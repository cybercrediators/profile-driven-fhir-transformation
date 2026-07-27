import pytest

from controller.pipeline_controller.pipeline_matchbox_sync import PipelineMatchboxSync


pytestmark = pytest.mark.unit


def make_sync(make_app_state, state, dataIO, matchbox, cache=None):
    app_state = make_app_state(dataIO=dataIO, cache=cache, conf={"profile_path": "p"})
    return PipelineMatchboxSync(app_state, state, matchbox)


# --------------------------------------------------------------------------- #
# _ensure_resource decision matrix
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "present, kwargs, expect_status, expect_uploads",
    [
        ("read_only present", dict(read_only=True, force_upload=False), "present", 0),
        ("read_only absent", dict(read_only=True, force_upload=False), "abort", 0),
        (
            "upload already no force",
            dict(read_only=False, force_upload=False, is_uploaded=True),
            "skipped",
            0,
        ),
        (
            "upload already force",
            dict(read_only=False, force_upload=True, is_uploaded=True),
            "uploaded",
            1,
        ),
        (
            "upload not-yet",
            dict(read_only=False, force_upload=False, is_uploaded=False),
            "uploaded",
            1,
        ),
    ],
)
def test_ensure_resource_matrix(
    make_app_state, state, fake_dataio, fake_matchbox, present, kwargs, expect_status, expect_uploads
):
    is_present = "present" in present
    mb = fake_matchbox(present_urls=["u"] if is_present else [])
    sync = make_sync(make_app_state, state, fake_dataio(), mb)

    uploads = []
    status = sync._ensure_resource(
        "StructureMap",
        "u",
        payload_loader=lambda: {"r": 1},
        uploader=lambda p: uploads.append(p),
        label="SM",
        **kwargs,
    )
    assert status == expect_status
    assert len(uploads) == expect_uploads


def test_ensure_resource_presence_default_skip(make_app_state, state, fake_dataio, fake_matchbox):
    """When is_uploaded is omitted, presence in matchbox decides the skip."""
    mb = fake_matchbox(present_urls=["u"])
    sync = make_sync(make_app_state, state, fake_dataio(), mb)
    uploads = []
    status = sync._ensure_resource(
        "StructureMap", "u", read_only=False, force_upload=False,
        payload_loader=lambda: {"r": 1}, uploader=lambda p: uploads.append(p), label="SM",
    )
    assert status == "skipped" and uploads == []


def test_ensure_resource_no_payload(make_app_state, state, fake_dataio, fake_matchbox):
    mb = fake_matchbox(present_urls=[])
    sync = make_sync(make_app_state, state, fake_dataio(), mb)
    uploads = []
    status = sync._ensure_resource(
        "StructureMap", "u", read_only=False, force_upload=True,
        payload_loader=lambda: None, uploader=lambda p: uploads.append(p), label="SM",
    )
    assert status == "no_payload" and uploads == []


# --------------------------------------------------------------------------- #
# check_matchbox_connection
# --------------------------------------------------------------------------- #


def test_check_matchbox_connection_ok(make_app_state, state, fake_dataio, fake_matchbox):
    sync = make_sync(make_app_state, state, fake_dataio(), fake_matchbox(capability=True))
    assert sync.check_matchbox_connection() is True


def test_check_matchbox_connection_unreachable(make_app_state, state, fake_dataio, fake_matchbox):
    sync = make_sync(make_app_state, state, fake_dataio(), fake_matchbox(capability=False))
    assert sync.check_matchbox_connection() is False


# --------------------------------------------------------------------------- #
# prepare_matchbox_setup
# --------------------------------------------------------------------------- #


def _full_project(write_json):
    profile = write_json("profile.json", {"resourceType": "StructureDefinition", "url": "http://x/p"})
    helper = write_json("helper.json", {"resourceType": "StructureDefinition", "url": "http://x/h"})
    sm = write_json("sm.json", {"resourceType": "StructureMap", "url": "http://x/sm"})
    cm = write_json("cm.json", {"resourceType": "ConceptMap", "url": "http://x/cm"})
    return profile, helper, sm, cm


def test_prepare_aborts_when_matchbox_unreachable(make_app_state, state, fake_dataio, fake_matchbox):
    sync = make_sync(make_app_state, state, fake_dataio(), fake_matchbox(capability=False))
    assert sync.prepare_matchbox_setup() is False


def test_prepare_uploads_everything_on_empty_server(
    make_app_state, state, fake_dataio, fake_matchbox, write_json
):
    profile, helper, sm, cm = _full_project(write_json)
    dataIO = fake_dataio(
        profile_files=[profile], helper_files=[helper], sm_files=[sm], cm_files=[cm]
    )
    mb = fake_matchbox(present_urls=[], ig_installed=False)
    sync = make_sync(make_app_state, state, dataIO, mb)

    assert sync.prepare_matchbox_setup() is True
    # IG installed + every resource uploaded once
    assert mb.installed_packages, "IG package should be installed when missing"
    uploaded_types = sorted(t for t, _ in mb.uploaded)
    assert uploaded_types == ["ConceptMap", "StructureDefinition", "StructureDefinition", "StructureMap"]
    # helper url discovered from disk and recorded in shared state
    assert "http://x/h" in state.source_helper_urls


def test_prepare_read_only_passes_when_all_present(
    make_app_state, state, fake_dataio, fake_matchbox, write_json
):
    profile, helper, sm, cm = _full_project(write_json)
    dataIO = fake_dataio(
        profile_files=[profile], helper_files=[helper], sm_files=[sm], cm_files=[cm]
    )
    mb = fake_matchbox(
        present_urls=["http://x/p", "http://x/h", "http://x/sm", "http://x/cm"],
        ig_installed=True,
    )
    sync = make_sync(make_app_state, state, dataIO, mb)

    assert sync.prepare_matchbox_setup(read_only=True) is True
    assert mb.uploaded == []  # read-only never uploads


def test_prepare_read_only_fails_when_resource_missing(
    make_app_state, state, fake_dataio, fake_matchbox, write_json
):
    profile, helper, sm, cm = _full_project(write_json)
    dataIO = fake_dataio(profile_files=[profile], helper_files=[helper], sm_files=[sm], cm_files=[cm])
    # profile present, but everything else missing -> abort at first missing
    mb = fake_matchbox(present_urls=["http://x/p"], ig_installed=True)
    sync = make_sync(make_app_state, state, dataIO, mb)

    assert sync.prepare_matchbox_setup(read_only=True) is False
    assert mb.uploaded == []


def test_prepare_skips_unchanged_profile_then_force_reuploads(
    make_app_state, state, fake_dataio, fake_matchbox, write_json
):
    # Full project so prepare reaches the end; helper+sm already present so they don't
    # interfere — we assert specifically on the profile (url http://x/p), which is absent.
    profile, helper, sm, cm = _full_project(write_json)
    dataIO = fake_dataio(profile_files=[profile], helper_files=[helper], sm_files=[sm], cm_files=[cm])
    mb = fake_matchbox(
        present_urls=["http://x/h", "http://x/sm", "http://x/cm"], ig_installed=True
    )
    sync = make_sync(make_app_state, state, dataIO, mb)

    def profile_uploads():
        return sum(
            1 for t, p in mb.uploaded if isinstance(p, dict) and p.get("url") == "http://x/p"
        )

    # first run: profile absent -> uploaded + hash recorded
    assert sync.prepare_matchbox_setup() is True
    assert profile_uploads() == 1

    # second run: unchanged hash -> profile skipped
    mb.uploaded.clear()
    assert sync.prepare_matchbox_setup() is True
    assert profile_uploads() == 0

    # force_upload -> profile re-uploaded despite unchanged hash
    mb.uploaded.clear()
    assert sync.prepare_matchbox_setup(force_upload=True) is True
    assert profile_uploads() == 1


# --------------------------------------------------------------------------- #
# sync_changed_resources
# --------------------------------------------------------------------------- #


def test_sync_changed_resources_reuploads_only_changed(
    make_app_state, state, fake_dataio, fake_matchbox, write_json
):
    sm = write_json("sm.json", {"resourceType": "StructureMap", "url": "http://x/sm"})
    dataIO = fake_dataio(sm_files=[sm])
    mb = fake_matchbox()
    sync = make_sync(make_app_state, state, dataIO, mb)

    # Establish a baseline hash for the current content.
    sync._record_upload_hashes()
    assert sync.sync_changed_resources() is False  # nothing changed

    # Mutate the file -> hash differs -> re-upload.
    sm.write_text('{"resourceType": "StructureMap", "url": "http://x/sm", "v": 2}')
    state.structure_map_urls = ["stale"]
    assert sync.sync_changed_resources() is True
    assert ("StructureMap", {"resourceType": "StructureMap", "url": "http://x/sm", "v": 2}) in mb.uploaded
    # changed SMs reset the loaded url list so it rebuilds from disk next time
    assert state.structure_map_urls == []


def test_hash_round_trip(make_app_state, state, fake_dataio, fake_matchbox):
    sync = make_sync(make_app_state, state, fake_dataio(), fake_matchbox())
    sync._save_upload_hashes({"a": "1"})
    assert sync._load_upload_hashes() == {"a": "1"}


def test_load_hashes_missing_returns_empty(make_app_state, state, fake_dataio, fake_matchbox):
    sync = make_sync(make_app_state, state, fake_dataio(), fake_matchbox())
    assert sync._load_upload_hashes() == {}
