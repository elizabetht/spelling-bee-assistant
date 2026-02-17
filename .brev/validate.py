#!/usr/bin/env python3
"""
Brev Configuration Validator for Spelling Bee Assistant

This script validates that all required files and configurations are present
for a successful Brev deployment.
"""

import os
import sys
import yaml
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'

def check_mark(success):
    return f"{Colors.GREEN}✓{Colors.RESET}" if success else f"{Colors.RED}✗{Colors.RESET}"

def validate_brev_config():
    """Validate .brev.yaml configuration file"""
    print("\n=== Validating Brev Configuration ===\n")
    
    # Check if .brev.yaml exists
    config_path = Path('.brev.yaml')
    if not config_path.exists():
        print(f"{check_mark(False)} .brev.yaml not found")
        return False
    print(f"{check_mark(True)} .brev.yaml exists")
    
    # Load and validate YAML
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"{check_mark(True)} .brev.yaml is valid YAML")
    except yaml.YAMLError as e:
        print(f"{check_mark(False)} .brev.yaml has YAML syntax errors: {e}")
        return False
    
    # Validate required fields
    required_fields = {
        'name': 'Application name',
        'description': 'Application description',
        'app': 'App configuration',
        'environment': 'Environment variables',
        'resources': 'Resource requirements'
    }
    
    all_valid = True
    for field, description in required_fields.items():
        if field in config:
            print(f"{check_mark(True)} {description} present")
        else:
            print(f"{check_mark(False)} Missing required field: {field}")
            all_valid = False
    
    # Validate app section
    if 'app' in config:
        app = config['app']
        if 'port' in app:
            print(f"{check_mark(True)} Port configured: {app['port']}")
        if 'start_command' in app:
            print(f"{check_mark(True)} Start command: {app['start_command']}")
    
    # Validate environment variables
    if 'environment' in config:
        env = config['environment']
        if 'required' in env and len(env['required']) > 0:
            print(f"{check_mark(True)} Required env vars defined: {len(env['required'])}")
            for var in env['required']:
                print(f"  - {var['name']}: {var.get('description', 'No description')}")
        else:
            print(f"{Colors.YELLOW}⚠{Colors.RESET} No required environment variables defined")
    
    return all_valid

def validate_brev_directory():
    """Validate .brev directory and its contents"""
    print("\n=== Validating .brev Directory ===\n")
    
    brev_dir = Path('.brev')
    if not brev_dir.exists():
        print(f"{check_mark(False)} .brev directory not found")
        return False
    print(f"{check_mark(True)} .brev directory exists")
    
    # Check required files
    required_files = {
        'setup.sh': 'Setup script for dependencies',
        'README.md': 'Brev-specific documentation'
    }
    
    all_valid = True
    for filename, description in required_files.items():
        filepath = brev_dir / filename
        if filepath.exists():
            print(f"{check_mark(True)} {filename} ({description})")
            
            # Check if setup.sh is executable
            if filename == 'setup.sh':
                if os.access(filepath, os.X_OK):
                    print(f"{check_mark(True)} setup.sh is executable")
                else:
                    print(f"{Colors.YELLOW}⚠{Colors.RESET} setup.sh is not executable (will be fixed during deployment)")
        else:
            print(f"{check_mark(False)} Missing: {filename}")
            all_valid = False
    
    # Check optional files
    optional_files = ['QUICKSTART.md', 'launch-example.yaml']
    for filename in optional_files:
        filepath = brev_dir / filename
        if filepath.exists():
            print(f"{check_mark(True)} {filename} (optional)")
    
    return all_valid

def validate_application_files():
    """Validate core application files"""
    print("\n=== Validating Application Files ===\n")
    
    required_files = {
        'spelling_bee_agent_backend.py': 'Main application',
        'requirements.txt': 'Python dependencies',
        'README.md': 'Main documentation',
        'Dockerfile': 'Container configuration'
    }
    
    all_valid = True
    for filename, description in required_files.items():
        filepath = Path(filename)
        if filepath.exists():
            print(f"{check_mark(True)} {filename} ({description})")
        else:
            print(f"{check_mark(False)} Missing: {filename}")
            all_valid = False
    
    # Check optional but important files
    optional_files = {
        'ui/index.html': 'Web UI',
        'guardrails/config.yml': 'NeMo Guardrails config',
        'patch_nvidia_pipecat.py': 'Pipecat patches'
    }
    
    for filename, description in optional_files.items():
        filepath = Path(filename)
        if filepath.exists():
            print(f"{check_mark(True)} {filename} ({description})")
        else:
            print(f"{Colors.YELLOW}⚠{Colors.RESET} Optional file missing: {filename}")
    
    return all_valid

def validate_readme_brev_section():
    """Check if README.md has Brev documentation"""
    print("\n=== Validating README.md ===\n")
    
    readme_path = Path('README.md')
    if not readme_path.exists():
        print(f"{check_mark(False)} README.md not found")
        return False
    
    try:
        content = readme_path.read_text()
        
        # Check for Brev section
        has_brev_section = 'brev' in content.lower() or 'launch with brev' in content.lower()
        
        if has_brev_section:
            print(f"{check_mark(True)} README.md contains Brev documentation")
        else:
            print(f"{Colors.YELLOW}⚠{Colors.RESET} README.md does not mention Brev deployment")
        
        return has_brev_section
    except Exception as e:
        print(f"{check_mark(False)} Error reading README.md: {e}")
        return False

def main():
    """Run all validations"""
    print("=" * 60)
    print("Brev Configuration Validator")
    print("Spelling Bee Assistant")
    print("=" * 60)
    
    results = {
        'config': validate_brev_config(),
        'directory': validate_brev_directory(),
        'application': validate_application_files(),
        'readme': validate_readme_brev_section()
    }
    
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60 + "\n")
    
    all_passed = all(results.values())
    
    for check, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        color = Colors.GREEN if passed else Colors.RED
        print(f"{color}{status}{Colors.RESET}: {check.capitalize()} validation")
    
    print("\n" + "=" * 60)
    
    if all_passed:
        print(f"{Colors.GREEN}✓ All validations passed!{Colors.RESET}")
        print("\nYour application is ready to be launched with Brev.")
        print("Next steps:")
        print("  1. Set ELEVENLABS_API_KEY secret in Brev")
        print("  2. Run: brev launch")
        print("  3. Access your application at the provided URL")
        print("\nSee .brev/QUICKSTART.md for detailed instructions.")
        return 0
    else:
        print(f"{Colors.RED}✗ Some validations failed{Colors.RESET}")
        print("\nPlease fix the issues above before launching with Brev.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
