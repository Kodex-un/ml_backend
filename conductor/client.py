import argparse
import logging
import os

import grpc

logger = logging.getLogger('classifier')


import protocol_pb2
import service_pb2
import service_pb2_grpc


parser = argparse.ArgumentParser()
parser.add_argument('--addr', default='0.0.0.0:1523', type=str, help='address:port to connect to')
parser.add_argument('message', type=str)
FLAGS = parser.parse_args()

def main():
    #np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    with grpc.insecure_channel(FLAGS.addr) as channel:
        stub = service_pb2_grpc.ClassifierStub(channel)
        res = stub.TextClassification(request=protocol_pb2.TextClassificationRequest(
            id='random id string',
            batch=[FLAGS.message],
        ))
        if res.error_code != 0:
            logger.info(f'response: {res.id}: error_code: {res.error_code}, status: {res.error_status}')
        else:
            logger.info(f'response: {res.id}: results: {res.results}')


if __name__ == '__main__':
    main()
