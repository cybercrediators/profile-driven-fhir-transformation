import json
import os
import shutil
import tarfile

from enum import Enum
from pathlib import Path
import logging
logger = logging.getLogger(__name__)
from typing import Union
from helpers import utils
from data_handling.instance_validation import is_definition_resource  # noqa: F401 – re-exported


class DataIO:
    """Class for handling all file and folder operations within the project directory."""
    def __init__(self, project_dir: Path):
        self.project_dir = Path(project_dir).resolve()
        logger.info(
            "Using project folder: %s. Checking if project structure already exists and creating folders...",
            self.project_dir,
        )
        self.project_dir.mkdir(exist_ok=True)
        self.standard_folders = [folder.value for folder in self.ProjectFolders]
        self.generate_project_structure()
        self.project_name = self.project_dir.name

    class ProjectFolders(Enum):
        PROCESSED_RESOURCES = "processed_resources"
        SOURCE_DATA = "source_data"
        CONCEPT_MAPS = "source_data/concept_maps"
        STRUCTURE_MAPS = "structure_maps"
        INPUT_PROFILE = "input_profile"
        SOURCE_MAPS = "source_definitions"
        EXAMPLES = "examples"

    def create_folder(self, folder_path: str):
        """Creates a new folder within the project directory."""
        path = self.project_dir / folder_path
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clear_project_folders(self):
        """Clears files from the standard project folders"""
        for folder in self.standard_folders:
            dir_path = self.project_dir / folder

            if not dir_path.exists():
                continue

            logger.info("Clearing project folders (OVERWRITE!)")
            for item in dir_path.iterdir():
                try:
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
                except Exception as e:
                    logger.error("Error deleting file %s: %s", item, e)

    def generate_project_structure(self):
        """Creates the project structure with predefined folders."""
        # print(f"Project folder will be: {self.project_dir}")
        for folder in self.standard_folders:
            self.create_folder(folder)
        logger.info("Project folder is ready to use.")

    def delete_folder(self, folder_path: str):
        """Deletes a folder and all its contents."""
        path = self.project_dir / folder_path
        if path.exists() and path.is_dir():
            shutil.rmtree(path)

    def update_folder_name(self, old_folder_path: str, new_folder_path: str):
        """Renames a folder."""
        old_path = self.project_dir / old_folder_path
        new_path = self.project_dir / new_folder_path
        if old_path.exists() and old_path.is_dir():
            os.rename(old_path, new_path)

    def read_file(self, file_path: str, is_binary: bool = False):
        """Reads and returns the content of a file."""
        path = self.project_dir / file_path
        if not path.exists() or not path.is_file():
            logger.error(f"Error: File '{path}' not found.")
            return None
        if is_binary:
            with open(path, "rb") as f:
                return f.read()
        with open(path, "r", encoding="UTF-8") as f:
            return f.read()

    def update_file(
        self, file_path: str, content: Union[str, bytes], append: bool = True
    ):
        """Updates a file by appending or overwriting content."""
        path = self.project_dir / file_path
        if not path.exists() or not path.is_file():
            logger.error(f"Error: File '{path}' not found.")
            return
        mode = "ab" if append else "wb"
        if not isinstance(content, bytes):
            mode = "a" if append else "w"
        with open(path, mode) as f:
            f.write(content)
        logger.info(f"File updated: {path}")

    def rename_file(self, old_file_path: str, new_file_path: str):
        """Renames a file."""
        old_path = self.project_dir / old_file_path
        new_path = self.project_dir / new_file_path
        if not old_path.exists() or not old_path.is_file():
            logger.error(f"Error: File '{old_path}' not found.")
            return

        # Ensure parent directory of new path exists
        new_path.parent.mkdir(parents=True, exist_ok=True)
        os.rename(old_path, new_path)
        logger.info(f"File renamed to: {new_path}")

    def delete_file(self, file_path: str):
        path = self.project_dir / file_path
        if not path.exists() or not path.is_file():
            logger.error(f"Error: File '{path}' not found.")
            return
        os.remove(path)
        logger.info(f"File deleted: {path}")

    def check_processed_resources(self):
        """Check if project resources already exist"""
        return self._check_existing_maps(self.ProjectFolders.PROCESSED_RESOURCES)

    def check_processed_helper_maps(self):
        """Check if processed helper maps already exist"""
        return self._check_existing_maps(self.ProjectFolders.SOURCE_MAPS)

    def check_processed_structure_maps(self):
        """Check if processed structure maps already exist"""
        return self._check_existing_maps(self.ProjectFolders.STRUCTURE_MAPS)

    def _check_existing_maps(self, directory: ProjectFolders):
        folder_dir = self.project_dir / directory.value
        if not folder_dir.is_dir():
            return False
        return any(folder_dir.iterdir())

    def store_project_file(
        self,
        project_folder: ProjectFolders,
        filename: str,
        content,
        mode="JSON",
        overwrite: bool = False,
    ):
        """Stores a json object in a file in the specified project folder."""
        file_path = self.project_dir / project_folder.value / filename

        if file_path.exists() and not overwrite:
            logger.warning(
                f"File '{file_path}' already exists and overwrite is set to False. Skipping."
            )
            return

        if mode == "JSON":
            utils.store_json(content, file_path)
        else:
            utils.store_json_str(content, file_path)
        logger.info(f"File stored: {file_path}")

    def project_file_exists(
        self, project_folder: ProjectFolders, filename: str
    ) -> bool:
        """Return True if a file exists in the given project folder."""
        return (self.project_dir / project_folder.value / filename).is_file()

    def load_project_file(self, project_folder: ProjectFolders, filename: str):
        """Loads a json object from a file in the specified project folder."""
        file_path = self.project_dir / project_folder.value / filename

        if not file_path.exists() or not file_path.is_file():
            logger.error(f"File '{file_path}' not found.")
            return None

        res_json = utils.get_json(file_path)
        logger.info(f"File loaded: {file_path}")
        return res_json

    def load_project_files(self, project_folder: ProjectFolders):
        """Loads all json objects from files in the specified project folder."""
        folder_dir = self.project_dir / project_folder.value
        if not folder_dir.is_dir():
            logger.error(f"Folder '{folder_dir}' not found.")
            return []

        loaded_files = []
        for file in folder_dir.iterdir():
            if file.suffix == ".json":
                res_json = utils.get_json(file)
                loaded_files.append((file.name, res_json))
                logger.info(f"File loaded: {file}")

        return loaded_files

    def get_filenames(self, path):
        # logger.info(f"Loading files from directory: {path}")
        return list(Path(path).iterdir())

    def get_json_profile_files(self, path):
        profile_dir = self.project_dir / self.ProjectFolders.INPUT_PROFILE.value
        if path:
            profile_dir = Path(path)
            logger.info("Using custom profile path: %s", profile_dir)
        logger.info(f"Get all json files from directory: {profile_dir}")
        return filter(lambda x: x.suffix == ".json", self.get_filenames(profile_dir))

    def add_tarred_profile(self, output_path: Path, overwrite=False):
        # Untar and load a given fhir profile
        profile_dir = self.project_dir / self.ProjectFolders.INPUT_PROFILE.value

        # check if profile files in input folder already exist
        if any(profile_dir.iterdir()):
            if overwrite:
                logger.info(f"Overwriting existing profile files in '{profile_dir}'.")
            else:
                logger.warning(
                    f"Warning: Profile files in '{profile_dir}' already exist. Skipping."
                )
                return

        # check if tar file exists
        if not output_path.exists() or not output_path.is_file():
            logger.error(f"Error: Profile '{output_path}' could not be found.")
            return

        # untar the profile
        with tarfile.open(output_path, "r") as tar_ref:
            tar_ref.extractall(profile_dir)

    def find_tarred_profile(self, tarred_profile_path: str = None):
        # Try to find a tarred profile in the given path or project folder
        if tarred_profile_path:
            tar_path = Path(tarred_profile_path)
            if tar_path.exists() and tar_path.is_file():
                return tar_path
            else:
                logger.error(
                    f"Error: Provided tarred profile '{tarred_profile_path}' could not be found."
                )
                return None

        # check project folder for tarred profile
        profile_dir = self.project_dir
        for file in profile_dir.iterdir():
            if file.suffix in [".tar", ".tgz"]:
                logger.info(f"Found tarred profile: {file}")
                return file

        logger.info("No tarred profile found in project folder.")
        return None

    def create_tar_package(
        self,
        input_files: Path = None,
        output_path: Path = None,
        overwrite: bool = False,
    ):
        """Create a tgz package using all *.json files from the input profile folder, OR use given input/output paths"""
        if output_path is None:
            output_path = self.project_dir / f"{self.project_name}.tgz"

        # check if tar package already exists
        tar_pkg = self.find_tarred_profile()
        if tar_pkg and not overwrite:
            logger.info(
                f"Tarred profile package already exists: {tar_pkg}. Skipping package creation."
            )
            return tar_pkg
        if tar_pkg and overwrite:
            logger.info("Overwriting existing tarred profile package!")
            self.delete_file(tar_pkg)

        # check if package.json exists in profile folder; generate a minimal one if absent
        package_json_path = (
            self.project_dir / self.ProjectFolders.INPUT_PROFILE.value / "package.json"
        )
        if not package_json_path.exists() or not package_json_path.is_file():
            minimal_pkg = {
                "name": self.project_name.lower().replace(" ", "-"),
                "version": "0.1.0",
                "type": "fhir.ig.package",
                "description": f"Auto-generated package for {self.project_name}",
                "fhirVersions": ["4.0.1"],
            }
            package_json_path.write_text(json.dumps(minimal_pkg, indent=2))
            logger.info(f"Generated minimal package.json for {self.project_name}")

        # create package from profile folder (i.e. just tar the input folder), must be in the `package/` dir inside the tar!
        with tarfile.open(output_path, "w:gz") as tar:
            package_dir = tarfile.TarInfo(name="package")
            package_dir.type = tarfile.DIRTYPE
            # TODO: rework actual permissions
            package_dir.mode = 0o755
            tar.addfile(package_dir)
            for p_file in self.get_json_profile_files(input_files):
                tar.add(p_file, arcname=f"package/{p_file.name}")
        return tar

    def access_package_file(self, package_file_name: str = "package.json"):
        """Access a given file from the project folder package.json file"""
        package_file = (
            self.project_dir
            / self.ProjectFolders.INPUT_PROFILE.value
            / package_file_name
        )
        if not package_file.exists() or not package_file.is_file():
            logger.error(f"Package file '{package_file}' not found.")
            return None
        return utils.get_json(package_file)

    def get_package_name(self, package_file_name: str = "package.json"):
        """Try to get the package name from the project folder package.json file"""
        package_json = self.access_package_file(package_file_name)
        if package_json is None:
            return None
        return package_json.get("name")

    def get_package_version(self, package_file_name: str = "package.json"):
        """Try to get the package version from the project folder package.json file"""
        package_json = self.access_package_file(package_file_name)
        if package_json is None:
            return None
        return package_json.get("version")

    def get_structure_map_files(self):
        """Get all StructureMap JSON file paths from the structure_maps folder."""
        return self._get_json_files_from_folder(self.ProjectFolders.STRUCTURE_MAPS)

    def get_concept_map_files(self):
        """Get all ConceptMap JSON file paths from the source_data/concept_maps folder."""
        return self._get_json_files_from_folder(self.ProjectFolders.CONCEPT_MAPS)

    def get_source_helper_map_files(self):
        """Get all source helper StructureDefinition JSON file paths from the source_definitions folder."""
        return self._get_json_files_from_folder(self.ProjectFolders.SOURCE_MAPS)

    def get_processed_resources(self):
        """Get all JSON file paths from the processed_resources/ folder."""
        return self._get_json_files_from_folder(self.ProjectFolders.PROCESSED_RESOURCES)

    def get_example_files(self):
        """Get all json files dropped into the examples/ folder (manual instance examples)."""
        return self._get_json_files_from_folder(self.ProjectFolders.EXAMPLES)

    def _get_json_files_from_folder(self, folder: ProjectFolders):
        """Get all json files from a given project folder"""
        json_files = []
        folder_dir = self.project_dir / folder.value
        for file in folder_dir.iterdir():
            if file.suffix == ".json":
                json_files.append(file)
        return json_files
