"""Pinput Shaping Plugin for OctoPrint
Perform input shaping tests on 3D printers using a accelerometers
"""

import csv
import inspect
import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, Optional

import flask
import numpy as np
import octoprint.plugin
import octoprint.plugin.core
import octoprint.printer
import octoprint.printer.profile
import octoprint.settings
import pexpect
from octoprint.logging.handlers import CleaningTimedRotatingFileHandler

from .inputshaping_analyzer import InputShapingAnalyzer


class PinputShapingPlugin(octoprint.plugin.StartupPlugin, # pylint: disable=too-many-ancestors,too-many-instance-attributes
                          octoprint.plugin.EventHandlerPlugin,
                          octoprint.plugin.ProgressPlugin,
                          octoprint.plugin.SimpleApiPlugin,
                          octoprint.plugin.SettingsPlugin,
                          octoprint.plugin.AssetPlugin,
                          octoprint.plugin.TemplatePlugin
                          ):
    """Pinput Shaping Plugin Main Class"""

    _printer: octoprint.printer.PrinterInterface
    _printer_profile_manager: octoprint.printer.profile.PrinterProfileManager
    _settings: octoprint.settings.Settings
    _plugin_manager: octoprint.plugin.core.PluginManager
    _identifier: str

    csv_filename: str
    metadata_dir: str
    current_axis: Optional[str] = None
    shapers: Dict[str, Dict[str, float]]

    def __init__(self) -> None:
        """Initialize the plugin."""

        super().__init__()
        self.plugin_data_folder = self.get_plugin_data_folder()
        self.metadata_dir = os.path.join(self.plugin_data_folder, "metadata")
        self.graphs_dir = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "static", "metadata"
        )
        self.csv_filename = None
        self.accelerometer_capture_active = False
        self._adchild = None
        self.current_axis = None
        self.shapers = {}
        self.get_m593 = False

        self._plugin_logger = logging.getLogger(f"octoprint.plugins.{__plugin_name__}")

    def configure_logger(self) -> None:
        """Configure the plugin logger."""

        log_base_path = os.path.expanduser("~/.octoprint/logs")

        # Create the directory if it doesn't exist
        if not os.path.exists(log_base_path):
            os.makedirs(log_base_path, exist_ok=True)
            os.chmod(log_base_path, 0o775)

        log_file_path = os.path.join(log_base_path, f"{__plugin_name__}.log")
        handler = CleaningTimedRotatingFileHandler(
            log_file_path, when="D", backupCount=3)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s: %(message)s"))
        self._plugin_logger.addHandler(handler)
        self._plugin_logger.propagate = False

    def get_current_function_name(self) -> str:
        """Get the name of the current function."""

        frame = inspect.currentframe()
        if frame is not None and frame.f_back is not None:
            return inspect.getframeinfo(frame.f_back).function
        # If we can't determine the function name, log a warning and return a default value
        self._plugin_logger.warning("Could not determine calling function name.")
        return "unknown_function"

    def get_settings_defaults(self) -> dict:
        """Return the default settings for the plugin."""

        printer_profile = self._printer_profile_manager.get_current_or_default()
        width = printer_profile['volume']['width']
        depth = printer_profile['volume']['depth']
        height = printer_profile['volume']['height']
        return {
            "sizeX": width,
            "sizeY": depth,
            "sizeZ": height,
            "accelMin": 300,
            "accelMax": 2500,
            "freqStart": 5,
            "freqEnd": 132,
            "dampingRatio": "0.05",
            "sensorType": "adxlspi"
        }

    def get_template_configs(self) -> list[dict]:
        """Return the template configurations for the plugin."""

        return [
            dict(type="settings", template="settings_pinput_shaping_settings.jinja2",
                 name="Pinput Shaping", custom_bindings=True),
            dict(type="tab", template="pinput_shaping_tab.jinja2", name="Input Shaping",
                 custom_bindings=True)
        ]

    def get_assets(self) -> dict:
        """Return the assets (JavaScript, CSS) for the plugin."""

        return {
            "js": ["js/pinput_shaping.js"] # JS file
        }

    def get_assets_folder(self) -> str:
        """Return the path to the plugin's static assets folder."""

        return os.path.join(os.path.dirname(os.path.realpath(__file__)), "static")

    def on_after_startup(self) -> None:
        """Called after the plugin has started."""

        self.configure_logger()
        self._plugin_logger.info(">>>>>> PInput-Shaping Loaded <<<<<<")
        self._plugin_logger.info("Plugin identifier is: %s", self._identifier)
        self._plugin_logger.info("Plugin version is: %s", self._plugin_version)
        self._plugin_logger.info("X size: %s", self._settings.get(['sizeX']))
        self._plugin_logger.info("Y size: %s", self._settings.get(['sizeY']))
        self._plugin_logger.info("Z size: %s", self._settings.get(['sizeZ']))
        self._plugin_logger.info("Acceleration min: %s", self._settings.get(['accelMin']))
        self._plugin_logger.info("Acceleration max: %s", self._settings.get(['accelMax']))
        self._plugin_logger.info("Frequency start: %s", self._settings.get(['freqStart']))
        self._plugin_logger.info("Frequency end: %s", self._settings.get(['freqEnd']))
        self._plugin_logger.info("Damping ratio: %s", self._settings.get(['dampingRatio']))
        self._plugin_logger.info("Sensor type: %s", self._settings.get(['sensorType']))

        self._plugin_manager.send_plugin_message(
            self._identifier, {"msg": "Pinput Shaping Plugin loaded"}
        )

        # Create the directory if it doesn't exist
        os.makedirs(self.metadata_dir, exist_ok=True)
        os.chmod(self.metadata_dir, 0o775)
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.chmod(self.graphs_dir, 0o775)

        self._plugin_logger.info(
            ">>>>>> PInput-Shaping Metadata directory initialized: %s", self.metadata_dir)
        self._plugin_logger.info(
            ">>>>>> PInput-Shaping Graphs directory initialized: %s", self.graphs_dir)

    def get_api_commands(self) -> Optional[dict]: # type: ignore
        """Return the API commands for the plugin."""

        return dict(run_accelerometer_test=[],
                    run_resonance_test=[])

    def on_api_command(self, command: str, data: dict) -> flask.Response:
        """Handle API commands sent to the plugin."""

        self._plugin_logger.info(
            ">>>>>> PInput-Shaping API Command: %s with data: %s", command, data)
        if command == "run_accelerometer_test":
            self._plugin_manager.send_plugin_message(
                self._identifier,
                {"type": "popup", "message": "Running Test for accelerometer..."},
            )
            result = self._run_accelerometer_test()
            return flask.jsonify(result)

        if command == "run_resonance_test":
            axis = data["data"]["axis"]
            x = data["data"]["start_x"]
            y = data["data"]["start_y"]
            z = data["data"]["start_z"]
            self._plugin_logger.info("Triggering resonance test for axis %s", axis)
            self._plugin_manager.send_plugin_message(
                self._identifier,
                {
                    "type": "popup",
                    "message": f"Running Resonance Test for {axis} Axis...",
                },
            )
            result = self._run_resonance_test(axis, x, y, z)
            return flask.jsonify(result)

        self._plugin_logger.warning("Unknown API command: %s", command)
        return flask.jsonify({"success": False, "error": "Unknown command"})

    def _run_accelerometer_test(self) -> dict:
        """Run the accelerometer test and return the results."""

        self._plugin_logger.info(">>>>>>> Running accelerometer test with settings:")
        self._plugin_logger.info("X size: %s", self._settings.get(['sizeX']))
        self._plugin_logger.info("Y size: %s", self._settings.get(['sizeY']))
        self._plugin_logger.info("Z size: %s", self._settings.get(['sizeZ']))
        self._plugin_logger.info("Acceleration min: %s", self._settings.get(['accelMin']))
        self._plugin_logger.info("Acceleration max: %s", self._settings.get(['accelMax']))
        self._plugin_logger.info("Frequency start: %s", self._settings.get(['freqStart']))
        self._plugin_logger.info("Frequency end: %s", self._settings.get(['freqEnd']))
        self._plugin_logger.info("Damping ratio: %s", self._settings.get(['dampingRatio']))
        self._plugin_logger.info("Sensor type: %s", self._settings.get(['sensorType']))

        try:
            self._plugin_logger.info("Backing up current shaper values...")
            self._printer.commands("M593")
            time.sleep(2)
            self.csv_filename = os.path.join(self.metadata_dir, "accelerometer_test_capture.csv")
            log_filename = os.path.join(self.metadata_dir, "accelerometer_output.log")

            self._start_accelerometer_capture(5)

            time.sleep(2)

            self._stop_accelerometer_capture()

            if not os.path.exists(self.csv_filename):
                self._plugin_logger.error("CSV data file not found")
                self._plugin_logger.error("Accelerometer test failed. No data captured.")
                my_err = "CSV data file not found"
                self._plugin_manager.send_plugin_message(
                    self._identifier, {"type": "error_popup", "message": my_err}
                )
                return {"success": False, "error": my_err}

            samples = []
            with open(self.csv_filename, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    samples.append({
                        "time": row["time"],
                        "x": row["x"],
                        "y": row["y"],
                        "z": row["z"]
                    })

            summary_line = "No summary available"
            if os.path.exists(log_filename):
                with open(log_filename, "r", encoding="utf-8") as logf:
                    lines = logf.read().strip().splitlines()
                    for line in reversed(lines):
                        if "samples" in line and "Hz" in line:
                            summary_line = line
                            break
            else:
                self._plugin_logger.warning("Output log not found")

            self._plugin_logger.info(
                "Accelerometer test completed. Summary: %s", summary_line
            )
            self._plugin_manager.send_plugin_message(
                self._identifier, dict(type="close_popup")
            )
            self.restore_shapers()
            return {
                "success": True,
                "summary": summary_line,
                "samples": samples,
                "stdout_preview": "\n".join(
                    [f"{s['time']} {s['x']} {s['y']} {s['z']}" for s in samples[-5:]]
                )  # last few samples
            }

        except Exception as e: # pylint: disable=broad-exception-caught
            self._plugin_logger.error("Accelerometer test failed: %s", e)
            self._plugin_manager.send_plugin_message(
                self._identifier, {"type": "error_popup", "message": str(e)}
            )
            return {"success": False, "error": str(e)}

    def _run_resonance_test(self, axis, x, y, z) -> dict:
        """Run the resonance test for the specified axis at given coordinates."""

        self._plugin_logger.info("Running resonance test for %s axis at position (%s, %s, %s)",
                                 axis, x, y, z)
         #create variable with the value of datetime in iso format
        dt= time.strftime("%Y%m%dT%H%M%S")
        self.csv_filename = os.path.join(self.metadata_dir,
                                         f"Raw_accel_values_AXIS_{axis}_{dt}.csv")

        printer_status = self._printer.get_state_id()

        if printer_status == "OPERATIONAL":
            self._plugin_logger.info("Printer is idle. Proceeding with resonance test.")
            self._printer.commands(f"M118 {__plugin_name__}: Store Shapers")
            self._printer.commands("M593")
            time.sleep(2)
            self._plugin_logger.info("Sending resonance test commands to printer...")
            self.home_and_park(x, y, z)
            self._printer.commands(self.precompute_sweep(axis, x, y))
            return {
                "success": True,
                "summary": f"Resonance test for {axis} triggered successfully."
            }

        self._plugin_manager.send_plugin_message(
            self._identifier, dict(type="close_popup")
        )
        time.sleep(1)
        message = f"Printer is not idle. State: {printer_status}. Cannot run resonance test."
        self._plugin_manager.send_plugin_message(
            self._identifier, {"type": "error_popup", "message": message}
        )
        self._plugin_logger.warning(message)
        return {"success": False, "error": message}

    def precompute_sweep(self, axis, x, y) -> list:
        """Precompute the resonance test commands for the specified axis."""

        num_cycles = 800
        steps_per_cycle = 4

        amplitude = 5
        min_amp = 1
        self.current_axis = axis

        accel_min = int(self._settings.get(["accelMin"]))
        accel_max = int(self._settings.get(["accelMax"]))

        amplitudes = np.linspace(amplitude, min_amp, num_cycles)
        accelerations = np.linspace(accel_min, accel_max, num_cycles)
        feedrates = np.clip(100 * accelerations, 2000, 15000)

        commands = []
        commands.append("M117 Starting resonance test")
        commands.append(f"M118 {__plugin_name__}: Accelerometer|ON")
        commands.append("M593 F0")
        commands.append(f"M117 Resonance Test on {axis}-Axis")

        current_accel = int(accelerations[0])
        commands.append(f"M204 S{current_accel}")

        for i in range(num_cycles):
            amp = amplitudes[i]
            accel = int(accelerations[i])
            feed = int(feedrates[i])

            if abs(accel - current_accel) > 100:
                commands.append(f"M204 S{accel}")
                current_accel = accel

            for j in range(steps_per_cycle):
                phase = 2 * np.pi * j / steps_per_cycle
                offset = amp * np.sin(phase)

                if axis == "X":
                    commands.append(f"G0 X{x + offset:.3f} Y{y:.3f} F{feed}")
                elif axis == "Y":
                    commands.append(f"G0 X{x:.3f} Y{y + offset:.3f} F{feed}")

        commands.append(f"M118 {__plugin_name__}: Resonance Test complete")
        commands.append("M204 P1500 R500 T1500")  # restoring original accel
        commands.append("M400")  # Wait for all moves to complete

        return commands

    def home_and_park(self, x, y, z) -> None:
        """Home and park the printer at the specified coordinates."""

        self._plugin_logger.info("Homing and parking printer...")
        start_pos = f"X{x} Y{y} Z{z}"
        self._printer.commands("G28")
        self._printer.commands(f"G0 {start_pos} F1500")
        self._printer.commands("G4 P1000")

    def gcode_received_handler(self, _comm, line, *_args, **_kwargs) -> str:
        """Handle received G-code lines and process Input Shaping commands."""

        if f"{__plugin_name__}:Store Shapers" in line:
            self._plugin_logger.info("Detected M118: Store Shapers message")
            self.get_m593 = True
            self.shapers = {}

        # Extract the shaper values from the line
        match_x = re.match(r".*M593 X F([\d.]+) D([\d.]+)", line)
        match_y = re.match(r".*M593 Y F([\d.]+) D([\d.]+)", line)

        if match_x and self.get_m593:
            self._plugin_logger.info("Detected M593: X value")
            self.shapers["X"] = {
                "F": float(match_x.group(1)),
                "D": float(match_x.group(2))
            }
        if match_y and self.get_m593:
            self._plugin_logger.info("Detected M593: Y value")
            self.shapers["Y"] = {
                "F": float(match_y.group(1)),
                "D": float(match_y.group(2))
            }
            # Save to file
            shaper_bck_path = os.path.join(
                self.metadata_dir, "current_shaper_values.json"
            )
            with open(shaper_bck_path, "w", encoding="utf-8") as f:
                json.dump(self.shapers, f)
            self._plugin_logger.info("Shaper backup saved: %s", self.shapers)
            self.get_m593 = False

        elif f"{__plugin_name__}: Resonance Test complete" in line:
            self._plugin_logger.info("Detected M118: Resonance Test complete message")
            self._plugin_logger.info(
                "Resonance Test complete for %s axis", self.current_axis
            )
            self._plugin_logger.info("Stopping accelerometer capture...")
            threading.Thread(target=self._stop_accelerometer_capture).start()
            self._plugin_logger.info("Starting Input Shaping analysis...")
            self._plugin_manager.send_plugin_message(
                self._identifier,
                {"type": "popup", "message": "Starting Input Shaping analysis..."},
            )
            time.sleep(3)
            self.get_input_shaping_results()

        elif f"{__plugin_name__}: Accelerometer|ON" in line:
            self._plugin_logger.info("Detected M118: Start accelerometer capture")
            self._plugin_logger.info("Accelerometer capture started...")
            threading.Thread(target=self._start_accelerometer_capture(3200)).start()
        return line

    def restore_shapers(self) -> None:
        """Restore the saved shaper values from the backup file."""

        backup_path = os.path.join(self.metadata_dir, "current_shaper_values.json")
        if not os.path.exists(backup_path):
            self._plugin_logger.warning("No saved shaper settings found to restore.")
            return

        with open(backup_path, "r", encoding="utf-8") as f:
            shapers = json.load(f)

        for axis, settings in shapers.items():
            freq = settings.get("F")
            damp = settings.get("D")
            if freq is not None and damp is not None:
                cmd = f"M593 {axis} F{freq:.2f} D{damp} "
                self._printer.commands(cmd)
                self._plugin_logger.info("Restored: %s", cmd)
        self._plugin_logger.info("Restored shaper values to printer.")

    def get_input_shaping_results(self) -> dict:
        """Get the Input Shaping results after accelerometer capture."""

        # Ensure current_axis is set before proceeding
        if self.current_axis is None:
            self._plugin_logger.error("current_axis is not set. Cannot get Input Shaping results.")
            self._plugin_manager.send_plugin_message(
                self._identifier,
                {"type": "error_popup", "message": "Current axis not set for analysis."}
            )
            return {"success": False, "error": "Current axis not set"}

        self._plugin_logger.info(
            "Getting Input Shaping results for %s Axis...", self.current_axis
        )

        if self.accelerometer_capture_active:
            self._plugin_logger.warning(
                "Accelerometer capture is still active. Stopping it first."
            )
            self._stop_accelerometer_capture()

        if not os.path.exists(self.csv_filename):
            self._plugin_logger.error("CSV data file not found")
            self._plugin_manager.send_plugin_message(
                self._identifier, dict(type="close_popup")
            )
            time.sleep(1)
            self._plugin_manager.send_plugin_message(
                self._identifier,
                {"type": "error_popup", "message": "CSV data file not found"},
            )
            return {"success": False, "error": "CSV data file not found"}

        analyzer = InputShapingAnalyzer(
            self.graphs_dir,
            self.csv_filename,
            float(self._settings.get(["dampingRatio"])),
            100,
            self.current_axis,
            logger=self._plugin_logger,
        )
        best_shaper = analyzer.analyze()
        signal_path, psd_path, shaper_results, best_shaper, base_freq = analyzer.generate_graphs()
        command = analyzer.get_recommendation()
        data_for_plotly = analyzer.get_plotly_data()

        self._plugin_logger.info("Best shaper for %s axis: %s", self.current_axis, best_shaper)
        self._plugin_logger.info("Signal graph saved to: %s", signal_path)
        self._plugin_logger.info("PSD graph saved to: %s", psd_path)
        self._plugin_logger.info("Recommended command: %s", command)
        self._plugin_logger.info("Input Shaping analysis completed.")
        self._plugin_manager.send_plugin_message(self._identifier, dict(type="close_popup"))
        self._printer.commands(
            f"M117 Freq for {self.current_axis}:{base_freq:.2f}"
            f"Damp:{self._settings.get(['dampingRatio'])}"
        )
        self._plugin_manager.send_plugin_message(self._identifier, {
            "type": "results_ready",
            "msg": "Input Shaping analysis completed",
            "axis": self.current_axis.upper(),
            "best_shaper": str(best_shaper),
            "signal_path": str(signal_path),
            "psd_path": str(psd_path),
            "command": str(command),
            "csv_path": str(self.csv_filename),
            "results": {
                k: {
                    "vibr": float(v["vibr"]),
                    "accel": float(v["accel"]),
                } for k, v in shaper_results.items()
            },
            "base_freq": float(base_freq)
        })

        data_for_plotly.update({
            "type": "plotly_data",
            "description": "Input Shaping Plotly Data",
            "axis": self.current_axis.upper()
        })
        self._plugin_manager.send_plugin_message(self._identifier, data_for_plotly)
        self.restore_shapers()
        return {"success": True}

    def _start_accelerometer_capture(self, freq=3200) -> None:
        """Start the accelerometer capture process using pexpect."""

        wrapper = None
        sensor_type = self._settings.get(['sensorType'])
        plugin_dir = os.path.dirname(os.path.abspath(__file__))

        if sensor_type == 'lis2dw-usb':
            self._plugin_logger.info("Starting LIS2DW USB capture...")
            wrapper = os.path.join(plugin_dir, "lis2dw-usb")
            if freq == 5:
                self._plugin_logger.warning(
                    "LIS2DW sensor does not support 5Hz frequency. Test will run at minimum 200Hz."
                )
                freq = 200
            else:
                self._plugin_logger.info(
                    "LIS2DW sensor does not support frequency %sHz. Test will run at max 1600Hz.",
                      freq
                )
                freq = 1600

        elif sensor_type == 'adxl345-usb':
            self._plugin_logger.info("Starting ADXL345 USB capture...")
            wrapper = os.path.join(plugin_dir, "adxl345-usb")

        elif sensor_type == 'adxl345-i2c':
            self._plugin_logger.info("Starting ADXL345 I2C capture...")
            wrapper = os.path.join(plugin_dir, "adxl345-i2c")

        elif sensor_type == 'adxl345-spi':
            self._plugin_logger.info("Starting ADXL345 SPI capture...")
            wrapper = os.path.join(plugin_dir, "adxl345-spi")

        cmd = f"sudo {wrapper} -f {freq} -s {self.csv_filename}"
        logfile_path = os.path.join(os.path.dirname(self.csv_filename), "accelerometer_output.log")

        try:
            self._adchild = pexpect.spawn(cmd, timeout=600, encoding="utf-8")
            self._adchild.logfile = open(logfile_path, "w", encoding="utf-8")

            # Wait for the "Press Q to stop" prompt
            self._adchild.expect("Press Q to stop", timeout=600)
            self.accelerometer_capture_active = True
            self._plugin_logger.info("Accelerometer ready and capturing.")
        except pexpect.TIMEOUT:
            self._plugin_logger.error("Timed out waiting for accelerometer to start.")
            raise
        except pexpect.EOF:
            self._plugin_logger.error("Accelerometer process exited early. Check logs.")
            raise
        except Exception as e:
            self._plugin_logger.error("Unexpected error: %s", e)
            raise

    def _stop_accelerometer_capture(self) -> None:
        """Stop the accelerometer capture process and save the data."""

        self._plugin_logger.info("Stopping accelerometer capture...")
        if self._adchild and self._adchild.isalive():
            try:
                self._adchild.sendline("Q")
                self._adchild.expect("Saved .* samples", timeout=30)
                self._plugin_logger.info("Accelerometer confirmed data saved.")
            except pexpect.TIMEOUT:
                self._plugin_logger.warning("No save confirmation. Terminating...")
                self._adchild.terminate(force=True)
            except pexpect.EOF:
                self._plugin_logger.info("Process already exited.")
            finally:
                self.accelerometer_capture_active = False
                if self._adchild.logfile:
                    self._adchild.logfile.close()
        else:
            self.accelerometer_capture_active = False
            self._plugin_logger.warning("Process not alive.")

    def get_update_information(self) -> dict:
        """Return the update information for the plugin."""

        return {
            "Pinput_Shaping": {
                "displayName": "Pinput_Shaping Plugin",
                "displayVersion": self._plugin_version,
                # version check: github repository
                "type": "github_release",
                "user": "navaismo",
                "repo": "OctoPrint-Pinput_Shaping",
                "current": self._plugin_version,
                # update method: pip
                "pip":
                "https://github.com/navaismo/OctoPrint-Pinput_Shaping/archive/{target_version}.zip"
            }
        }

__plugin_name__ = "Pinput_Shaping"
__plugin_pythoncompat__ = ">=3,<4"  # Only Python 3


__plugin_implementation__: Optional[PinputShapingPlugin] = None
__plugin_hooks__: Optional[Dict[str, Any]] = None

def __plugin_load__() -> None:
    """Load the plugin when OctoPrint starts."""

    global __plugin_implementation__ # pylint: disable=global-statement

    __plugin_implementation__ = PinputShapingPlugin()

    global __plugin_hooks__ # pylint: disable=global-statement
    __plugin_hooks__ = {
        "octoprint.comm.protocol.gcode.received": __plugin_implementation__.gcode_received_handler
    }
