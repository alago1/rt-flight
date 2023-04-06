# RT-Flight

## Contributors

- [Allan Lago](https://github.com/alago1)
- [Sahaj Patel](https://github.com/sah4jpatel)
- [Matthew Clausen](https://github.com/matt-clausen)
- [Tom Liraz](https://github.com/tomliraz)
- [Tyler J. Schultz](https://github.com/tj-schultz)

## Before using

You may want to add a `/data` folder at the root with the appropriate images.

## To compile grpc protobuf
`cd raspberry_pi_code/protos && python -m grpc_tools.protoc -I../protos --python_out=. --pyi_out=. --grpc_python_out=. messaging.proto`