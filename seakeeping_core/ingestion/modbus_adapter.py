#!/usr/bin/env python3
"""
modbus_adapter.py — Modbus RTU/TCP Ingestion Adapter
=====================================================
Polls Modbus registers from ship PLCs and converts them into the same
canonical (field, value, status, native_quality) tuples consumed by
the Universal Data Bus.

Supports:
    - Modbus RTU (RS-485 serial) via pymodbus
    - Modbus TCP (Ethernet) via pymodbus

Register mappings are defined in the vessel YAML config under 'modbus_map'.
Each entry specifies the register address, data type, scaling factor,
and target canonical field.

YAML Example:
    modbus_map:
      engine_rpm:
        address: 40001
        register_type: "holding"
        data_type: "uint16"
        scale: 0.1
        offset: 0.0
        range: [0.0, 200.0]
      rudder:
        address: 40002
        register_type: "holding"
        data_type: "int16"
        scale: 0.1
        offset: 0.0
        range: [-35.0, 35.0]
"""

import time
from typing import Dict, List, Tuple, Optional

# pymodbus is optional — only import if Modbus is configured
try:
    from pymodbus.client import ModbusSerialClient, ModbusTcpClient
    PYMODBUS_AVAILABLE = True
except ImportError:
    PYMODBUS_AVAILABLE = False


class ModbusAdapter:
    """
    Polls Modbus registers and produces canonical update tuples
    compatible with UniversalShipDataBus.ingest_parsed_updates().
    """

    def __init__(self, modbus_config: dict, modbus_map: dict):
        """
        Args:
            modbus_config: Connection settings from YAML.
                For RTU: {'type': 'rtu', 'port': '/dev/ttyUSB0', 'baudrate': 9600,
                          'slave_id': 1}
                For TCP: {'type': 'tcp', 'host': '192.168.1.100', 'port': 502,
                          'slave_id': 1}
            modbus_map: Register-to-field mappings from YAML.
        """
        if not PYMODBUS_AVAILABLE:
            raise ImportError(
                "pymodbus is required for Modbus ingestion. "
                "Install with: pip install pymodbus"
            )

        self.modbus_map = modbus_map
        self.slave_id = modbus_config.get('slave_id', 1)
        self.client = None

        conn_type = modbus_config.get('type', 'tcp').lower()
        if conn_type == 'rtu':
            self.client = ModbusSerialClient(
                port=modbus_config['port'],
                baudrate=modbus_config.get('baudrate', 9600),
                timeout=modbus_config.get('timeout', 1.0),
            )
        elif conn_type == 'tcp':
            self.client = ModbusTcpClient(
                host=modbus_config['host'],
                port=modbus_config.get('port', 502),
                timeout=modbus_config.get('timeout', 1.0),
            )
        else:
            raise ValueError(f"Unknown Modbus connection type: {conn_type}")

    def connect(self) -> bool:
        """Establishes the Modbus connection."""
        return self.client.connect()

    def disconnect(self):
        """Closes the Modbus connection."""
        if self.client:
            self.client.close()

    def poll_all(self) -> List[Tuple[str, float, str, float]]:
        """
        Polls all configured Modbus registers and returns canonical tuples.

        Returns:
            List of (field_name, value, status, native_quality) tuples.
            Same format as NMEAParser.parse_line() output.
        """
        results = []
        timestamp = time.time()

        for field_name, spec in self.modbus_map.items():
            address = spec['address']
            reg_type = spec.get('register_type', 'holding')
            scale = spec.get('scale', 1.0)
            offset = spec.get('offset', 0.0)
            lo, hi = spec.get('range', [-9999.0, 9999.0])
            data_type = spec.get('data_type', 'uint16')

            try:
                # Read register based on type
                if reg_type == 'holding':
                    response = self.client.read_holding_registers(
                        address, count=1, slave=self.slave_id
                    )
                elif reg_type == 'input':
                    response = self.client.read_input_registers(
                        address, count=1, slave=self.slave_id
                    )
                else:
                    continue

                if response.isError():
                    results.append((field_name, 0.0, 'CRC_FAIL', 0.0))
                    continue

                raw_val = response.registers[0]

                # Handle signed integers
                if data_type == 'int16' and raw_val > 32767:
                    raw_val -= 65536

                # Apply scaling and offset
                converted = raw_val * scale + offset

                # Range validation
                status = 'VALID' if (lo <= converted <= hi) else 'OUT_OF_RANGE'

                # Modbus doesn't have native quality indicators like GPS HDOP
                # so native_quality defaults to 1.0
                results.append((field_name, converted, status, 1.0))

            except Exception:
                results.append((field_name, 0.0, 'CRC_FAIL', 0.0))

        return results
