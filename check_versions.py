import subprocess
import re


def get_latest_version(package_name):
    try:
        # Remove any comments from the package name
        package_name = package_name.split("#")[0].strip()
        # Skip empty lines
        if not package_name:
            return None, None, None

        # Extract package name without version
        name = package_name.split("==")[0].strip()
        current_version = (
            package_name.split("==")[1].strip() if "==" in package_name else None
        )

        # Run pip index versions command
        result = subprocess.run(
            ["pip", "index", "versions", name], capture_output=True, text=True
        )

        # Parse the output to get the latest version
        output = result.stdout
        latest_version_match = re.search(r"LATEST:\s+(\d+\.\d+\.\d+)", output)

        if latest_version_match:
            latest_version = latest_version_match.group(1)
            return name, latest_version, current_version
        else:
            # Try alternative regex pattern for packages without LATEST label
            version_match = re.search(name + r"\s+\(([0-9.]+)\)", output)
            if version_match:
                latest_version = version_match.group(1)
                return name, latest_version, current_version
            return name, None, current_version
    except Exception as e:
        print(f"Error checking {package_name}: {e}")
        return name, None, current_version


# Manual updates for packages we know
manual_updates = {
    "mamba-ssm": "2.2.4",
    "accelerate": "1.6.0",
    "bitsandbytes": "0.42.0",
    "peft": "0.15.2",
    "restrictedpython": "8.0",
    "trl": "0.17.0",
    "streamlit": "1.45.1",
}

# Read requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.readlines()

# Process each package
updates = []
for req in requirements:
    req = req.strip()
    # Skip comments and empty lines
    if req.startswith("#") or not req:
        updates.append(req)
        continue

    # Extract package name without version
    name = req.split("==")[0].strip()
    current_version = req.split("==")[1].strip() if "==" in req else None

    # Check if we have a manual update for this package
    if name in manual_updates:
        latest_version = manual_updates[name]
        if current_version and latest_version != current_version:
            print(f"Update available: {name} {current_version} -> {latest_version}")
            updates.append(f"{name}=={latest_version}")
        else:
            print(f"Up to date: {name} {current_version}")
            updates.append(req)
    else:
        # Get version info
        name, latest, current = get_latest_version(req)

        if name and latest:
            if current and latest != current:
                print(f"Update available: {name} {current} -> {latest}")
                updates.append(f"{name}=={latest}")
            elif current:
                print(f"Up to date: {name} {current}")
                updates.append(req)
            else:
                print(f"Adding version: {name} -> {latest}")
                updates.append(f"{name}=={latest}")
        else:
            updates.append(req)

# Generate updated requirements.txt
with open("requirements_updated.txt", "w") as f:
    f.write("\n".join(updates))

print("\nUpdated requirements saved to requirements_updated.txt")
