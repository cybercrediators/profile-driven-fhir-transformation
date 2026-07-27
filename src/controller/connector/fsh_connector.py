import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional
import docker
import docker.errors
from data_handling.data_io import DataIO
import logging
logger = logging.getLogger(__name__)

_CONTAINER_WORKDIR = "/opt/fsh_dir"
_CONTAINER_HOME = "/tmp/fsh-home"
_DEFAULT_DOCKER_IMAGE = "sushi-docker-sushi:latest"
_GENONCE_SCRIPT = "_genonce.sh"


class SushiController:
    """runs SUSHI to compile FSH projects and initialises bridge project folders"""

    def __init__(self, sushi_command: str = "sushi", docker_image: Optional[str] = None):
        self.sushi_command = sushi_command
        self.docker_image = docker_image
        self._docker_client = docker.from_env() if docker_image else None
        if not self.check_fsh_setup():
            if docker_image:
                logger.error(f"Docker image '{docker_image}' not found. Build it with: docker build sushi-docker/")
            else:
                logger.error(
                    f"No SUSHI setup found: default docker image '{_DEFAULT_DOCKER_IMAGE}' not available "
                    "(build with: docker compose build sushi-docker/) and no local fsh-sushi "
                    "(install with: npm install -g fsh-sushi)."
                )
            raise FileNotFoundError("SUSHI setup not found!")

    def check_fsh_setup(self) -> bool:
        """resolve a SUSHI setup: configured docker image > default docker image > verified local fsh-sushi"""
        if self.docker_image:
            # explicitly configured image: use it or fail loudly, never fall back
            return self._docker_image_exists(self.docker_image)
        return self._use_default_docker_image() or self._verify_local_sushi()

    def _use_default_docker_image(self) -> bool:
        """adopt the default sushi image when docker and the image are available"""
        try:
            client = docker.from_env()
            client.images.get(_DEFAULT_DOCKER_IMAGE)
        except docker.errors.DockerException as e:
            logger.debug(
                f"Default sushi docker image not usable ({type(e).__name__}: {e}), "
                f"trying local '{self.sushi_command}'"
            )
            return False
        self.docker_image = _DEFAULT_DOCKER_IMAGE
        self._docker_client = client
        logger.info(f"Using default SUSHI docker image '{_DEFAULT_DOCKER_IMAGE}'")
        return True

    def _verify_local_sushi(self) -> bool:
        """check the local binary really is fsh-sushi — /usr/bin/sushi may be GNOME Sushi (file previewer)"""
        binary = shutil.which(self.sushi_command)
        if binary is None:
            return False
        try:
            result = subprocess.run(
                [self.sushi_command, "--version"], capture_output=True, text=True, timeout=30
            )
        except (OSError, subprocess.SubprocessError) as e:
            logger.error(f"Could not run '{self.sushi_command} --version' ({type(e).__name__}: {e})")
            return False
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode == 0 and re.search(r"SUSHI v?\d", output, re.IGNORECASE):
            logger.warning(
                f"Using local '{binary}' ({output.strip().splitlines()[0]}) — version is unpinned, "
                "prefer the docker image for reproducible builds"
            )
            return True
        logger.error(f"'{binary}' is not fsh-sushi (name collision, e.g. GNOME Sushi?): {output.strip()[:200]}")
        return False

    def _docker_image_exists(self, image: str) -> bool:
        """check if the sushi docker image exists"""
        try:
            self._docker_client.images.get(image)  # type: ignore[union-attr]
            return True
        except docker.errors.ImageNotFound:
            return False

    @staticmethod
    def resolve_fsh_root(fsh_path: Path) -> Optional[Path]:
        """find the SUSHI project root (the dir holding sushi-config.yaml), walking up when a
        subfolder like <root>/input/fsh was passed; None when no root exists"""
        for candidate in (fsh_path, *list(fsh_path.parents)[:3]):
            if any((candidate / name).is_file() for name in ("sushi-config.yaml", "sushi-config.yml")):
                if candidate != fsh_path:
                    logger.info(
                        f"'{fsh_path}' is not the FSH project root — using '{candidate}' "
                        "(found sushi-config.yaml)"
                    )
                return candidate
        return None

    def _run_sushi(self, fsh_path: Path):
        """execute the sushi build command in the given directory, either via local installation or docker"""
        if self.docker_image:
            logs = self._docker_client.containers.run(  # type: ignore[union-attr]
                image=self.docker_image,
                command=["sushi", "build", "."],
                volumes={str(fsh_path): {"bind": _CONTAINER_WORKDIR, "mode": "rw"}},
                working_dir=_CONTAINER_WORKDIR,
                # run as the host user so fsh-generated/ isn't left root-owned on the mount
                user=f"{os.getuid()}:{os.getgid()}",
                remove=True,
                stderr=True,
            )
            if logs:
                logger.debug(logs.decode(errors="replace"))
        else:
            subprocess.run([self.sushi_command, "build", "."], cwd=fsh_path, check=True)

    def process_fsh(
        self,
        fsh_dir: str,
        project_path: str,
        project_name: Optional[str] = None,
        snapshots: bool = True,
    ):
        """compile an FSH project with SUSHI and populate input_profile/ in the bridge project"""
        fsh_path = Path(fsh_dir).resolve()
        if not fsh_path.exists() or not fsh_path.is_dir():
            logger.error(f"FSH directory not found: {fsh_path}")
            return

        logger.info(f"Running sushi in {fsh_path}...")
        try:
            self._run_sushi(fsh_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"Sushi execution failed: {e}")
            return
        except Exception as e:
            logger.error(f"Error running sushi: {e}")
            return

        if snapshots:
            self._run_genonce(fsh_path)

        source_files_dir = self._find_sushi_output(fsh_path)
        if not source_files_dir:
            logger.error(
                "Could not find generated JSON files in sushi output (checked fsh-generated/resources and output/)."
            )
            return

        if not project_name:
            project_name = fsh_path.name

        new_project_path = Path(project_path).resolve()
        logger.info(
            f"{'Updating' if new_project_path.exists() else 'Creating'} project at {new_project_path}"
        )

        DataIO(new_project_path)
        input_profile_dir = new_project_path / DataIO.ProjectFolders.INPUT_PROFILE.value

        for item in input_profile_dir.iterdir():
            if item.is_file():
                item.unlink()

        json_files = list(source_files_dir.glob("*.json"))
        for f in json_files:
            shutil.copy(f, input_profile_dir)

        if snapshots:
            self._overlay_snapshots(fsh_path, input_profile_dir)

        logger.info(
            f"FSH processing complete. {len(json_files)} files copied to {input_profile_dir}"
        )

    def _run_genonce(self, fsh_path: Path):
        """run the IG publisher for profile snapshots — sushi alone emits differentials only,
        which degrades map generation to stubs for profile-constrained elements"""
        script = fsh_path / _GENONCE_SCRIPT
        if not script.is_file():
            logger.warning(
                f"No {_GENONCE_SCRIPT} in the FSH project — skipping snapshot generation; "
                "StructureDefinitions stay differential-only (run with --no-snapshots to silence)."
            )
            return
        logger.info("Running IG publisher (genonce) for snapshots — this can take a few minutes...")
        try:
            if self.docker_image:
                self._run_genonce_docker(fsh_path)
            else:
                subprocess.run(["bash", _GENONCE_SCRIPT], cwd=fsh_path, check=True)
        except (subprocess.CalledProcessError, docker.errors.ContainerError) as e:
            logger.warning(
                f"IG publisher exited non-zero (tolerated — snapshots usually land in "
                f"temp/pages before the failing Jekyll step): {type(e).__name__}"
            )
        except Exception as e:
            logger.error(f"Error running IG publisher: {type(e).__name__}: {e}")

    def _run_genonce_docker(self, fsh_path: Path):
        """genonce inside the sushi image (ships java + jekyll); reuse the host FHIR package cache"""
        volumes = {str(fsh_path): {"bind": _CONTAINER_WORKDIR, "mode": "rw"}}
        fhir_cache = Path.home() / ".fhir"
        if fhir_cache.is_dir():
            volumes[str(fhir_cache)] = {"bind": f"{_CONTAINER_HOME}/.fhir", "mode": "rw"}
        logs = self._docker_client.containers.run(  # type: ignore[union-attr]
            image=self.docker_image,  # type: ignore[arg-type]
            command=["bash", _GENONCE_SCRIPT],
            volumes=volumes,
            working_dir=_CONTAINER_WORKDIR,
            environment={"HOME": _CONTAINER_HOME},
            user=f"{os.getuid()}:{os.getgid()}",
            remove=True,
            stderr=True,
        )
        if logs:
            logger.debug(logs.decode(errors="replace"))

    def _overlay_snapshots(self, fsh_path: Path, input_profile_dir: Path):
        """replace copied differential-only StructureDefinitions with the snapshot-bearing
        publisher output from temp/pages (same recipe as scripts/e2e_kfdm_smoke.sh)"""
        pages_dir = fsh_path / "temp" / "pages"
        upgraded, without_snapshot = 0, []
        for target in sorted(input_profile_dir.glob("StructureDefinition-*.json")):
            candidate = pages_dir / target.name
            if self._has_snapshot(candidate):
                shutil.copy(candidate, target)
                upgraded += 1
            elif not self._has_snapshot(target):
                without_snapshot.append(target.name)
        if upgraded:
            logger.info(f"Snapshots: {upgraded} StructureDefinitions upgraded from IG publisher output")
        if without_snapshot:
            logger.warning(
                f"{len(without_snapshot)} StructureDefinitions have no snapshot "
                f"(map generation degrades to stubs for these): {without_snapshot}"
            )

    @staticmethod
    def _has_snapshot(path: Path) -> bool:
        if not path.is_file():
            return False
        try:
            return "snapshot" in json.loads(path.read_text())
        except (OSError, ValueError) as e:
            logger.warning(f"Unreadable StructureDefinition candidate {path}: {type(e).__name__}: {e}")
            return False

    def _find_sushi_output(self, fsh_path: Path) -> Optional[Path]:
        """check common sushi output directories for generated JSON files"""
        for candidate in [
            fsh_path / "fsh-generated" / "resources",
            fsh_path / "output",
        ]:
            if candidate.is_dir() and list(candidate.glob("*.json")):
                return candidate
        return None
