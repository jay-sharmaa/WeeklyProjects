import os
import re
import glob
from pathlib import Path

def parse_gradle_dependencies(gradle_file_path):
    """Extract dependencies from a Gradle build file."""
    dependencies = []
    
    with open(gradle_file_path, 'r') as file:
        content = file.read()
        
        dependency_blocks = re.findall(r'dependencies\s*{([^}]*)}', content, re.DOTALL)
        
        for block in dependency_blocks:
            impl_deps = re.findall(r'implementation\s+[\'"]([^\'"]*)[\'"]\s*', block)
            impl_deps2 = re.findall(r'implementation\s*\(\s*[\'"]([^\'"]*)[\'"]\s*\)', block)
            api_deps = re.findall(r'api\s+[\'"]([^\'"]*)[\'"]\s*', block)
            
            dependencies.extend(impl_deps)
            dependencies.extend(impl_deps2)
            dependencies.extend(api_deps)
            
    return dependencies

def parse_dependency_string(dep_string):
    """Parse a dependency string into group, artifact, and version."""
    parts = dep_string.split(':')
    if len(parts) >= 3:
        return {
            'group': parts[0],
            'artifact': parts[1],
            'version': parts[2]
        }
    return None

def find_dependency_in_gradle_cache(dependency, gradle_cache_dir):
    """Find a dependency in the Gradle cache and get its size."""
    group_path = dependency['group'].replace('.', '/')
    artifact = dependency['artifact']
    version = dependency['version']
    
    jar_pattern = os.path.join(gradle_cache_dir, group_path, artifact, version, f"{artifact}-{version}*.jar").replace("\\", "/")
    aar_pattern = os.path.join(gradle_cache_dir, group_path, artifact, version, f"{artifact}-{version}*.aar").replace("\\", "/")
    
    print(f"Looking for JAR: {jar_pattern}")
    jar_files = glob.glob(jar_pattern)
    
    if not jar_files:
        print(f"Looking for AAR: {aar_pattern}")
        aar_files = glob.glob(aar_pattern)
        if aar_files:
            print(f"Found AAR file: {aar_files[0]}")
            return os.path.getsize(aar_files[0])

    if jar_files:
        print(f"Found JAR file: {jar_files[0]}")
        return os.path.getsize(jar_files[0])
    
    print(f"Could not find JAR or AAR for {artifact}-{version}")
    return 0

def estimate_installed_size(file_size):
    """Estimate the installed size of a dependency based on its compressed size."""
    return file_size * 1.5

def estimate_contribution_to_apk(file_size):
    """Estimate how much a dependency contributes to final APK size."""
    return file_size * 0.7

def estimate_app_size(total_apk_contribution):
    """Estimate the total app size (APK) based on the contributions of all dependencies."""
    base_apk_size = 5 * 1024 * 1024

    total_size = base_apk_size + total_apk_contribution
    return total_size

def analyze_project_dependencies(project_path):
    gradle_file_path = os.path.join(project_path, "app", "build.gradle.kts")
    if not os.path.exists(gradle_file_path):
        print(f"Cannot find build.gradle at {gradle_file_path}")
        return []
    
    print(f"Analyzing dependencies in {gradle_file_path}...")
    
    gradle_cache_dir = os.path.expanduser("~/.gradle/caches/modules-2/files-2.1")
    if not os.path.exists(gradle_cache_dir):
        print(f"Gradle cache not found at {gradle_cache_dir}. Please run a Gradle sync first.")
        return []
    
    dependencies = parse_gradle_dependencies(gradle_file_path)
    
    results = []
    total_size = 0
    
    for dep_string in dependencies:
        dep = parse_dependency_string(dep_string)
        if not dep:
            print(f"Skipping malformed dependency: {dep_string}")
            continue
            
        size = find_dependency_in_gradle_cache(dep, gradle_cache_dir)
        
        if size > 0:
            installed_size = estimate_installed_size(size)
            apk_contribution = estimate_contribution_to_apk(size)
            
            results.append({
                'dependency': f"{dep['group']}:{dep['artifact']}:{dep['version']}",
                'size': size,
                'installed_size': installed_size,
                'apk_contribution': apk_contribution
            })
            
            total_size += apk_contribution
        else:
            print(f"Could not find size for: {dep['group']}:{dep['artifact']}:{dep['version']}")
    
    return results, total_size

def main():
    project_path = r'C:\androidprojectkotlin\UITutorial'
    
    results, total_size = analyze_project_dependencies(project_path)
    
    if not results:
        print("No dependency sizes could be determined. Make sure to run a Gradle sync first.")
        return
    
    results.sort(key=lambda x: x['size'], reverse=True)
    
    estimated_app_size = estimate_app_size(total_size)
    
    print("\nDependency Size Analysis:")
    print("-" * 100)
    print(f"{'Dependency':<60} {'Size (KB)':<12} {'Installed (KB)':<15} {'APK Impact (KB)':<15}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['dependency']:<60} {result['size']/1024:<12.2f} {result['installed_size']/1024:<15.2f} {result['apk_contribution']/1024:<15.2f}")
    
    print("-" * 100)
    print(f"Total estimated contribution to APK size: {total_size / 1024 / 1024:.2f} MB")
    print(f"Estimated total app size: {estimated_app_size / 1024 / 1024:.2f} MB")
    
if __name__ == "__main__":
    main()
