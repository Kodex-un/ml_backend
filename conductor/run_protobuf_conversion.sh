#!/bin/bash

python3 -m grpc_tools.protoc -Iproto --python_out=. --pyi_out=. --grpc_python_out=. proto/service.proto proto/protocol.proto
