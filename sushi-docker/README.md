# Sushi Docker
- Small Docker container for the CRDM lecture
- Uses the latest node base image, installs `fsh-sushi` (including a working publisher)
- Can be used to build Implementation guides based on the `input` files in the corresponding `fsh_shared/xyz` sub-directory
- Use the `fsh-shared` subdirectories to create your FSH profile (e.g. patient/input/patient.fsh)

## Build
- Build the container: `docker compose build`

## Usage
- Edit or place your input files in the `fsh_shared/<project_name>/input` folder
- Either use the provided helper scripts to execute the corresponding commands per project:
  - `run_build.sh patient` -> runs the sushi build command for the patient (switch with "tutorial" for the tutorial sub-directory)
  - `run_updatepublisher.sh patient` -> runs the `_updatePublisher` command for the patient (switch with "tutorial" for the tutorial sub-directory)
  - `run_genonce.sh patient` -> runs the `_genonce` command for the patient (switch with "tutorial" for the tutorial sub-directory)
- OR Use the container directly (keep the linked `fsh_shared` folder in mind)
    + `docker compose run --rm sushi bash -c "cd tutorial" && sushi build .` (for the tutorial project)
    + `docker compose run --rm sushi bash -c "cd tutorial && ./_updatePublisher.sh`
    + `docker compose run --rm sushi bash -c "cd tutorial && ./_genonce.sh`
- Clean build files using the `clean_shared.sh` script
