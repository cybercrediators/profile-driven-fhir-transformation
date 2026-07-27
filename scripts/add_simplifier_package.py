import argparse
import subprocess
import json
import sys

from pathlib import Path

# add a simplifier package using npm and generate an index.json
# npm command structure:
# npm --registry https://packages.simplifier.net install [package_name]@[version] --prefix [resource_path]

def install_package(package_name: str, prefix: Path, registry: str) -> None:
    """Install the given NPM FHIR package"""
    # proc = subprocess.Popen(['npm', '--registry', str(registry), 'install', str(package_name), '--prefix', str(prefix)], stdout=subprocess.PIPE, universal_newlines=True)
    command = ['npm', '--registry', str(registry), 'install', str(package_name), '--prefix', str(prefix)]
    try:
        subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
        )
        print("Package installed successfully!")
    except FileNotFoundError:
        print("Error: 'npm' command was not found. Install it!")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print("Error while installing the package!")
        print(f"Return code: {e.returncode}")
        print(e)
        sys.exit(1)

def generate_index_file(package_path: Path) -> None:
    """Generate the .index.json file for a downloaded simplifier package"""
    print("Creating index file...")
    index = { "files": [] }
    # files = [f for f in os.listdir(package_path) if os.path.isfile(os.path.join(package_path, f))]
    for f in package_path.iterdir():
        if f.is_file() and f.suffix == '.json':
            try:
                with open(f, 'r', encoding = "utf-8") as content:
                    data = json.load(content)
                    if url := data.get("url"):
                        index["files"].append({ "filename": f.name, "url": url })
            except json.JSONDecodeError:
                print(f"Skipping non-JSON file: {f.name}")
            except Exception as e:
                print(f"Error with file {f.name}: {e}")

    index_path = package_path / ".index.json"
    with open(index_path, 'w', encoding="utf-8") as content:
        json.dump(index, content, ensure_ascii=False, indent=4)
    print("Index file written!")

def main():
    """Install given FHIR simplifier packages using npm"""
    parser = argparse.ArgumentParser(description="Add npm FHIR packages to the local resource folder registry with index generation (if applicable).")

    parser.add_argument(
        '--package-name',
        type=str,
        required=True,
        help='The URL of the FHIR package.'
    )

    parser.add_argument(
        '--package-version',
        type=str,
        required=True,
        help='The version of the FHIR package.'
    )

    parser.add_argument(
        '--prefix',
        type=str,
        default=Path('./data/resource_cache/local_packages'),
        help='Local resource folder, (default: ../data/resource_cache/local_packages)'
    )

    parser.add_argument(
        '--overwrite',
        default=False,
        action='store_true',
        help='Overwrite an existing .index.json file regardless'
    )

    parser.add_argument(
        '--registry-url',
        type=str,
        default="https://packages.simplifier.net",
        help='The URL of the package registry. Uses: https://packages.simplifier.net as default'
    )

    args = parser.parse_args()
    package_name = f"{args.package_name}@{args.package_version}"
    package_dir = args.prefix

    package_dir.mkdir(parents=True, exist_ok=True)

    # install npm package
    install_package(package_name, package_dir, args.registry_url)
    # package_path = os.path.join(package_dir, "node_modules", str(args.package_name))
    package_path = package_dir / "node_modules" / args.package_name

    if not Path(package_path / ".index.json").exists() or args.overwrite:
        generate_index_file(package_path)

if __name__ == '__main__':
    main()
