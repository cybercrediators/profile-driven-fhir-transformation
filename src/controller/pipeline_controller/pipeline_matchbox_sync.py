import hashlib
import json
from pathlib import Path

import logging
logger = logging.getLogger(__name__)
from helpers import utils

from data_handling.app_state import AppState
from controller.pipeline_controller.pipeline_state import PipelineState


class PipelineMatchboxSync:
    """matchbox sync handler (validate, upload, and change detection for profiles/maps)"""

    def __init__(self, app_state: AppState, state: PipelineState, matchbox_controller):
        self.app_state = app_state
        self.state = state
        self.matchbox_controller = matchbox_controller

    def check_matchbox_connection(self):
        """check if matchbox server is reachable and contains required profiles/maps (read-only)"""
        logger.info("Validating matchbox server setup...")
        capability_statement = self.matchbox_controller.get_capability_statement()
        if capability_statement:
            logger.info("Matchbox server is reachable. Capability statement received.")
        else:
            logger.error("Failed to reach matchbox server.")
            return False
        return True

    def prepare_matchbox_setup(self, force_upload=False, read_only=False):
        """prepare the setup for matchbox server interaction (re-upload through force_upload or just non-destructive check)"""
        if not self.check_matchbox_connection():
            logger.error("Matchbox server validation failed!")
            return False

        pkg_name = self.app_state.dataIO.get_package_name()
        if not self.matchbox_controller.check_implementation_guide_installed(
            ig_id=pkg_name
        ):
            if read_only:
                logger.warning(
                    f"IG '{pkg_name}' not installed in matchbox — relying on the "
                    "individually uploaded resources (read-only check continues)."
                )
            else:
                logger.info(
                    f"Implementation guide '{pkg_name}' not found in matchbox. Installing package..."
                )
                self.matchbox_controller.install_npm_package(
                    package_name=pkg_name,
                    package_version=self.app_state.dataIO.get_package_version(),
                    package_path=str(self.app_state.dataIO.find_tarred_profile()),
                )

        # check hashes for comparison
        hashes = self._load_upload_hashes()
        for f in self.app_state.dataIO.get_json_profile_files(
            self.app_state.conf.get("profile_path", "")
        ):
            data = utils.get_json(f)
            if (
                not isinstance(data, dict)
                or data.get("resourceType") != "StructureDefinition"
            ):
                continue
            url = data.get("url")
            if not url:
                continue
            key = str(f)
            current_hash = self._hash_file(Path(f))
            status = self._ensure_resource(
                "StructureDefinition",
                url,
                read_only=read_only,
                force_upload=force_upload,
                is_uploaded=(hashes.get(key) == current_hash),
                payload_loader=lambda data=data: data,
                uploader=self.matchbox_controller.upload_structure_definition,
                label="profile StructureDefinition",
            )
            if status == "abort":
                return False
            if status == "uploaded":
                hashes[key] = current_hash
        if not read_only:
            self._save_upload_hashes(hashes)

        helper_defs_by_url = {}
        for f in self.app_state.dataIO.get_source_helper_map_files() or []:
            data = utils.get_json(f)
            url = data.get("url") if isinstance(data, dict) else None
            if url:
                helper_defs_by_url[url] = data
        if not self.state.source_helper_urls:
            logger.info(
                "No helper definition URLs defined to check in matchbox, trying to update!"
            )
            if not helper_defs_by_url:
                logger.warning(
                    "No helper definition URLs defined to check in matchbox!"
                )
                return False
            self.state.source_helper_urls = list(helper_defs_by_url.keys())

        def _load_helper_payload(helper_url):
            # Prefer the cache; fall back to the on-disk source definition and refresh it.
            sd = self.app_state.cache.get_resource_from_cache(helper_url)
            if sd is None:
                sd = helper_defs_by_url.get(helper_url)
                if sd is not None:
                    self.app_state.cache.add_resource_to_cache(sd)
            return sd

        # check source definitions
        for helper_url in self.state.source_helper_urls:
            status = self._ensure_resource(
                "StructureDefinition",
                helper_url,
                read_only=read_only,
                force_upload=force_upload,
                payload_loader=lambda u=helper_url: _load_helper_payload(u),
                uploader=self.matchbox_controller.upload_structure_definition,
                label="helper StructureDefinition",
            )
            if status == "abort":
                return False

        # rebuild url list when force_upload, so cache is up to date
        if not read_only and (force_upload or not self.state.structure_map_urls):
            if not (
                structure_map_files := self.app_state.dataIO.get_structure_map_files()
            ):
                logger.warning("No structure map files found!")
                return False
            structure_map_files = sorted(structure_map_files, key=lambda x: str(x))
            urls = []
            for f in structure_map_files:
                data = utils.get_json(f)
                if (
                    not isinstance(data, dict)
                    or data.get("resourceType") != "StructureMap"
                ):
                    continue
                url = data.get("url")
                if not url:
                    continue
                self.app_state.cache.add_resource_to_cache(
                    data
                )  # always refresh cache from disk
                urls.append(url)
            self.state.structure_map_urls = urls

        sm_urls_to_check = self.state.structure_map_urls
        if read_only and not sm_urls_to_check:
            sm_files = sorted(
                self.app_state.dataIO.get_structure_map_files() or [], key=str
            )
            sm_urls_to_check = [
                url
                for f in sm_files
                if isinstance(d := utils.get_json(f), dict)
                and d.get("resourceType") == "StructureMap"
                for url in [d.get("url")]
                if url
            ]

        # prepare structure maps
        for structure_map_url in sm_urls_to_check:
            status = self._ensure_resource(
                "StructureMap",
                structure_map_url,
                read_only=read_only,
                force_upload=force_upload,
                payload_loader=lambda u=structure_map_url: self.app_state.cache.get_resource_from_cache(
                    u
                ),
                uploader=self.matchbox_controller.upload_structure_map,
                label="StructureMap",
            )
            if status == "abort":
                return False

        # prepare concept maps
        for f in self.app_state.dataIO.get_concept_map_files() or []:
            data = utils.get_json(f)
            url = data.get("url")
            if not url:
                continue
            status = self._ensure_resource(
                "ConceptMap",
                url,
                read_only=read_only,
                force_upload=force_upload,
                payload_loader=lambda data=data: data,
                uploader=self.matchbox_controller.upload_concept_map,
                label="ConceptMap",
            )
            if status == "abort":
                return False

        if not read_only:
            self._record_upload_hashes()
        return True

    def _ensure_resource(
        self,
        fhir_type,
        url,
        *,
        read_only,
        force_upload,
        payload_loader,
        uploader,
        is_uploaded=None,
        label=None,
    ):
        """check if resource exists or uploaded to matchbox"""
        label = label or fhir_type
        if read_only:
            if not self.matchbox_controller.get_resource_by_url(fhir_type, url):
                logger.warning(
                    f"{label} {url} not found in matchbox (read-only check)."
                )
                return "abort"
            logger.info(f"{label} {url} present.")
            return "present"

        if is_uploaded is None:
            is_uploaded = bool(
                self.matchbox_controller.get_resource_by_url(fhir_type, url)
            )
        if not (force_upload or not is_uploaded):
            logger.info(f"{label} {url} found -> is ready.")
            return "skipped"

        payload = payload_loader()
        if payload is None:
            logger.error(
                f"{label} {url} not found in cache or on disk! Cannot upload."
            )
            return "no_payload"
        action = "Re-uploading" if force_upload else "Uploading"
        logger.info(f"{action} {label} {url}...")
        uploader(payload)
        return "uploaded"

    def _record_upload_hashes(self) -> None:
        """persists current file hashes after a successful prepare/upload"""
        hashes = self._load_upload_hashes()
        for f in self.app_state.dataIO.get_structure_map_files() or []:
            hashes[str(f)] = self._hash_file(f)
        for f in self.app_state.dataIO.get_concept_map_files() or []:
            hashes[str(f)] = self._hash_file(f)
        for f in self.app_state.dataIO.get_json_profile_files(
            self.app_state.conf.get("profile_path", "")
        ):
            hashes[str(f)] = self._hash_file(Path(f))
        self._save_upload_hashes(hashes)

    def _hash_file(self, path: Path) -> str:
        """create hash of file content for change detection"""
        return hashlib.md5(path.read_bytes()).hexdigest()

    def _load_upload_hashes(self) -> dict:
        """load persisted file hashes for change detection"""
        hash_file = self.app_state.dataIO.project_dir / "upload_hashes.json"
        if hash_file.exists():
            try:
                return json.loads(hash_file.read_text())
            except Exception as e:
                logger.warning(
                    "Could not read upload-hash file %s (%s: %s) — treating all resources as changed.",
                    hash_file, type(e).__name__, e,
                )
        return {}

    def _save_upload_hashes(self, hashes: dict) -> None:
        """store created file content hashes for change detection"""
        hash_file = self.app_state.dataIO.project_dir / "upload_hashes.json"
        hash_file.write_text(json.dumps(hashes, indent=2))

    def sync_changed_resources(self) -> bool:
        """check (compare hashes) and re-synchronize changed resources using matchbox"""

        sm_files = self.app_state.dataIO.get_structure_map_files() or []
        cm_files = self.app_state.dataIO.get_concept_map_files() or []
        sd_files = list(
            self.app_state.dataIO.get_json_profile_files(
                self.app_state.conf.get("profile_path", "")
            )
        )
        all_files = sorted(set(str(f) for f in sm_files + cm_files + sd_files))
        if not all_files:
            return False

        hashes = self._load_upload_hashes()
        changed = False

        for key in all_files:
            f = Path(key)
            current_hash = self._hash_file(f)
            if hashes.get(key) == current_hash:
                continue

            data = utils.get_json(f)
            if not isinstance(data, dict):
                continue
            resource_type = data.get("resourceType")
            url = data.get("url")
            if not url or resource_type not in (
                "StructureMap",
                "ConceptMap",
                "StructureDefinition",
            ):
                continue

            logger.info(f"{resource_type} {url} changed on disk — re-uploading...")
            if resource_type == "StructureMap":
                self.app_state.cache.add_resource_to_cache(data)
                self.matchbox_controller.upload_structure_map(data)
            elif resource_type == "ConceptMap":
                self.matchbox_controller.upload_concept_map(data)
            else:
                self.matchbox_controller.upload_structure_definition(data)

            hashes[key] = current_hash
            changed = True

        if changed:
            self._save_upload_hashes(hashes)
            self.state.structure_map_urls = (
                []
            )  # force rebuild from disk on next transform_data call

        return changed
