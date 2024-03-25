from concurrent import futures

import argparse
import grpc
import logging
import os

logger = logging.getLogger('classifier')


import protocol_pb2
import service_pb2
import service_pb2_grpc

from transformers import pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--addr', default='0.0.0.0:1523', type=str, help='address:port to listen on')
parser.add_argument('--workdir', default='workdir', type=str, help='Working directory')
parser.add_argument('--model_path', default='citizenlab/twitter-xlm-roberta-base-sentiment-finetunned', type=str, help='Model name and path')
FLAGS = parser.parse_args()

class TextClassificationServer(service_pb2_grpc.ClassifierServicer):
    def __init__(self):
        self.classifier = pipeline("text-classification", model=FLAGS.model_path, tokenizer=FLAGS.model_path)
        logger.info(f'Created classifier')

    def TextClassification(self, request: protocol_pb2.TextClassificationRequest, context):
        try:
            res = self.classifier(list(request.batch))
            results = list(map(lambda r:
                          protocol_pb2.TextClassificationResult(label=r['label'], score=r['score']),
                          res))
                
            return protocol_pb2.TextClassificationResponse(
                id=request.id,
                error_code=0,
                error_status='',
                results=results,
            )
        except Exception as e:
            logger.error(f'request: {request.id}: exception: {e}')
            return protocol_pb2.TextClassificationResponse(
                id=request.id,
                error_code=-1,
                error_status=e.__str__(),
                results=[],
            )
    
def main():
    #np.set_printoptions(formatter={'float': '{:0.4f}'.format, 'int': '{:4d}'.format}, linewidth=250, suppress=True, threshold=np.inf)

    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(fmt='%(asctime)s.%(msecs)03d: %(message)s', datefmt='%d/%m/%y %H:%M:%S')
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    os.makedirs(FLAGS.workdir, exist_ok=True)
    handler = logging.FileHandler(os.path.join(FLAGS.workdir, 'server.log'), 'a')
    handler.setFormatter(fmt)
    logger.addHandler(handler)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service_pb2_grpc.add_ClassifierServicer_to_server(TextClassificationServer(), server)
    server.add_insecure_port(FLAGS.addr)
    server.start()
    logger.info(f'Started listening on {FLAGS.addr}')
    server.wait_for_termination()

if __name__ == '__main__':
    main()
