#!/usr/bin/env python3
"""
Fixed SNAP Diagnostics - Corrected command flags
===============================================
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class SNAPDiagnostics:
    """SNAP installation diagnostics and debugging - FIXED VERSION"""
    
    def __init__(self):
        self.snap_home = os.environ.get('SNAP_HOME')
        self.gpt_cmd = self.get_gpt_command()
    
    def get_gpt_command(self) -> str:
        """Get GPT command for the operating system"""
        if not self.snap_home:
            raise RuntimeError("SNAP_HOME not set")
        
        if sys.platform.startswith('win'):
            return os.path.join(self.snap_home, 'bin', 'gpt.exe')
        else:
            return os.path.join(self.snap_home, 'bin', 'gpt')
    
    def run_snap_diagnostics(self):
        """Run comprehensive SNAP diagnostics with correct commands"""
        logger.info("="*60)
        logger.info("SNAP DIAGNOSTICS - FIXED VERSION")
        logger.info("="*60)
        
        # 1. Check SNAP_HOME
        logger.info(f"SNAP_HOME: {self.snap_home}")
        if not self.snap_home:
            logger.error("❌ SNAP_HOME environment variable not set!")
            return False
        
        if not os.path.exists(self.snap_home):
            logger.error(f"❌ SNAP_HOME directory does not exist: {self.snap_home}")
            return False
        
        logger.info(f"✓ SNAP_HOME exists: {self.snap_home}")
        
        # 2. Check GPT executable
        logger.info(f"GPT executable: {self.gpt_cmd}")
        if not os.path.exists(self.gpt_cmd):
            logger.error(f"❌ GPT executable not found: {self.gpt_cmd}")
            return False
        
        logger.info(f"✓ GPT executable exists: {self.gpt_cmd}")
        
        # 3. Check SNAP version (corrected command)
        self.check_snap_version()
        
        # 4. Check Java configuration
        self.check_java_config()
        
        # 5. Check available operators
        self.check_available_operators()
        
        # 6. Test simple GPT command
        return self.test_simple_gpt()
    
    def check_snap_version(self):
        """Check SNAP version information with correct command"""
        try:
            # Use -h instead of --version (SNAP doesn't support --version)
            result = subprocess.run([self.gpt_cmd, '-h'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Extract version info from help text
                lines = result.stdout.split('\n')
                version_line = None
                for line in lines:
                    if 'version' in line.lower() or 'snap' in line.lower():
                        version_line = line.strip()
                        break
                
                if version_line:
                    logger.info(f"✓ SNAP Info: {version_line}")
                else:
                    logger.info("✓ SNAP GPT help command works")
            else:
                logger.warning(f"⚠ GPT help command returned error: {result.stderr}")
        except Exception as e:
            logger.warning(f"⚠ Error checking SNAP version: {e}")
    
    def check_java_config(self):
        """Check Java configuration"""
        try:
            # Check system Java
            java_result = subprocess.run(['java', '-version'], 
                                       capture_output=True, text=True, timeout=10)
            if java_result.returncode == 0:
                java_version = java_result.stderr.split('\n')[0]  # Version is in stderr
                logger.info(f"✓ System Java: {java_version}")
            else:
                logger.warning("⚠ Could not detect system Java")
                
        except Exception as e:
            logger.warning(f"⚠ Error checking Java config: {e}")
    
    def check_available_operators(self):
        """Check available SNAP operators"""
        try:
            # List available operators using correct flag
            result = subprocess.run([self.gpt_cmd, '-e'], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                available_ops = result.stdout.lower()
                logger.info("✓ Available operators check:")
                
                # Check for specific operators we need
                operators_to_check = ['read', 'write', 's2resampling', 'subset', 'c2rcc.msi']
                
                for op in operators_to_check:
                    if op in available_ops:
                        logger.info(f"  ✓ {op}")
                    else:
                        logger.error(f"  ❌ {op} - NOT FOUND!")
                        
                # Check for problematic plugins
                if 'aster' in available_ops:
                    logger.warning("  ⚠ ASTER plugin detected - this may cause issues")
                
            else:
                logger.warning(f"⚠ Could not list operators: {result.stderr}")
                
        except Exception as e:
            logger.error(f"❌ Error checking operators: {e}")
    
    def test_simple_gpt(self):
        """Test a simple GPT command"""
        try:
            logger.info("Testing simple GPT command...")
            
            # Test GPT with help command
            result = subprocess.run([self.gpt_cmd, '-h'], 
                                  capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                logger.info("✓ Simple GPT test successful")
                
                # Check for plugin errors in output
                if "NoClassDefFoundError" in result.stderr:
                    logger.error("❌ Plugin errors detected in GPT output!")
                    logger.error("This will cause processing failures")
                    return False
                elif "WARNING" in result.stderr and "aster" in result.stderr.lower():
                    logger.warning("⚠ ASTER plugin warnings detected")
                    logger.warning("Consider disabling ASTER plugin in SNAP Desktop")
                
                return True
            else:
                logger.error(f"❌ Simple GPT test failed:")
                logger.error(f"  Return code: {result.returncode}")
                logger.error(f"  STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ GPT test timed out")
            return False
        except Exception as e:
            logger.error(f"❌ GPT test error: {e}")
            return False
    
    def suggest_fixes(self):
        """Suggest potential fixes for common SNAP issues"""
        logger.info("="*60)
        logger.info("SUGGESTED FIXES FOR SNAP ISSUES")
        logger.info("="*60)
        
        logger.info("🔧 IMMEDIATE FIXES:")
        logger.info("1. Start SNAP Desktop:")
        logger.info('   "C:\\Program Files\\esa-snap\\bin\\snap64.exe"')
        logger.info("   - Go to Tools > Plugins > Installed")
        logger.info("   - Disable or uninstall ASTER/EOMTBX plugins")
        logger.info("   - Restart SNAP")
        
        logger.info("2. Reset SNAP user directory:")
        logger.info("   rmdir /s \"%USERPROFILE%\\.snap\"")
        
        logger.info("3. Check Java version:")
        logger.info("   java -version")
        logger.info("   (Should be Java 8 or 11)")
        
        logger.info("🔧 IF PROBLEMS PERSIST:")
        logger.info("4. Reinstall SNAP:")
        logger.info("   - Download from https://step.esa.int/")
        logger.info("   - Uninstall current version first")
        logger.info("   - Install with default options")
        
        logger.info("5. Manual plugin cleanup:")
        logger.info("   - Delete: %USERPROFILE%\\.snap\\system\\modules")
        logger.info("   - Start SNAP Desktop to reinitialize")

def test_minimal_processing(input_file: str, output_dir: str):
    """Test minimal SNAP processing with better error detection"""
    
    diagnostics = SNAPDiagnostics()
    
    # Run diagnostics first
    if not diagnostics.run_snap_diagnostics():
        logger.error("SNAP diagnostics failed!")
        diagnostics.suggest_fixes()
        return False
    
    # Create minimal test graph
    graph_content = '''<?xml version="1.0" encoding="UTF-8"?>
<graph id="Minimal_Test">
  <version>1.0</version>
  
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${sourceProduct}</file>
    </parameters>
  </node>
  
  <node id="Write">
    <operator>Write</operator>
    <sources>
      <sourceProduct refid="Read"/>
    </sources>
    <parameters class="com.bc.ceres.binding.dom.XppDomElement">
      <file>${targetProduct}</file>
      <formatName>BEAM-DIMAP</formatName>
    </parameters>
  </node>
  
</graph>'''
    
    graph_file = 'minimal_test_fixed.xml'
    with open(graph_file, 'w', encoding='utf-8') as f:
        f.write(graph_content)
    
    try:
        # Test minimal processing
        output_file = os.path.join(output_dir, "test_minimal_fixed.dim")
        
        cmd = [
            diagnostics.gpt_cmd,
            graph_file,
            f'-PsourceProduct={input_file}',
            f'-PtargetProduct={output_file}',
            '-c', '2G',  # Reduced memory
            '-q', '1'    # Single thread
        ]
        
        logger.info("Testing minimal SNAP processing...")
        logger.info(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Minimal processing test successful!")
            if os.path.exists(output_file):
                file_size = os.path.getsize(output_file)
                logger.info(f"✓ Output file created: {file_size} bytes")
                # Cleanup
                os.remove(output_file)
                data_dir = output_file.replace('.dim', '.data')
                if os.path.exists(data_dir):
                    import shutil
                    shutil.rmtree(data_dir)
                return True
            else:
                logger.warning("⚠ Processing succeeded but no output file found")
                return False
        else:
            logger.error("❌ Minimal processing test failed!")
            logger.error(f"Return code: {result.returncode}")
            logger.error(f"STDERR: {result.stderr[:500]}...")
            
            # Check for specific errors
            if "NoClassDefFoundError" in result.stderr:
                logger.error("🚨 PLUGIN ERROR DETECTED!")
                logger.error("This is the same error causing your processing failures")
                diagnostics.suggest_fixes()
            
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Processing test timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Processing test error: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(graph_file):
            os.remove(graph_file)

if __name__ == "__main__":
    # Quick SNAP diagnostics
    import logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    diagnostics = SNAPDiagnostics()
    result = diagnostics.run_snap_diagnostics()
    
    if not result:
        diagnostics.suggest_fixes()
    else:
        logger.info("✅ SNAP diagnostics passed!")