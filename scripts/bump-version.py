from pathlib import Path

import click
import semver
from git import Repo

ROOT_DIR = Path(__file__).absolute().parent.parent
VERSION_FILE = ROOT_DIR / "version.txt"


@click.command()
@click.argument("level", type=click.Choice(("major", "minor", "patch")))
@click.option("--no-commit")
def main(level: str, no_commit: bool):
    with open(VERSION_FILE) as f:
        version = f.read().strip()
    print(f"Previous version: {version}")

    version = semver.VersionInfo.parse(version)

    if level == "major":
        version = version.bump_major()
    elif level == "minor":
        version = version.bump_minor()
    elif level == "patch":
        version = version.bump_patch()
    else:
        assert False

    print(f"New version: {version}")

    with open(VERSION_FILE, "w") as f:
        f.write(f"{version}\n")

    if not no_commit:
        print(f"Committing and tagging version {version}")

        repo = Repo(ROOT_DIR)
        index = repo.index
        remote = repo.remote("origin")

        # Commit and push the version change
        index.add([str(VERSION_FILE)])
        index.commit(f"Bump version to {version}")
        remote.push()

        # Create and push new version tag
        tag = repo.create_tag(version)
        remote.push(tag)


if __name__ == "__main__":
    main()
