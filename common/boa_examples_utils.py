# -*- coding: utf-8 -*-
#
# Licensed Materials - Property of IBM
# “Restricted Materials of IBM”
# 5765-R17
# © Copyright IBM Corp. 2020  All Rights Reserved.
# US Government Users Restricted Rights - Use, duplication or disclosure restricted
# by GSA ADP Schedule Contract with IBM Corp
#

"""
Utility methods for BOA Example Programs
"""

import argparse
import os

class BoaExamplesUtils:
    """
    Utility methods for BOA Example Programs
    """

    def __init__(self):
        pass

    @classmethod
    def validate_ca_cert_path(cls, ca_cert_path):
        """
        Validate ca_cert_path exists and is readable
        """
        ca_cert = None
        msg = "Path specified in ca_cert_path not a file or not readable"
        if os.path.isfile(ca_cert_path):
            try:
                with open(ca_cert_path, 'r', encoding='utf-8') as f:
                    ca_cert = f.read()
            except Exception as e:
                pass
        if ca_cert == None:
            raise argparse.ArgumentTypeError(msg)
        return ca_cert_path

    @classmethod
    def validate_port(cls, port_value):
        """
        Validate the port number in port_value
        """
        msg = "port must be an integer between 1 and 65535"
        value_is_valid = False
        try:
            port = int(port_value)
            if 1 <= port <= 65535:
                value_is_valid = True
        except ValueError:
           pass
        if not value_is_valid:
            raise argparse.ArgumentTypeError(msg)
        return port

    @classmethod
    def validate_epochs(cls, epochs_value):
        """
        Validate the number of epochs in epochs_value
        """
        msg = "epochs must be a positive integer"
        value_is_valid = False
        try:
            epochs = int(epochs_value)
            if epochs > 0:
                value_is_valid = True
        except ValueError:
           pass
        if not value_is_valid:
            raise argparse.ArgumentTypeError(msg)
        return epochs

    @classmethod
    def get_connection_url(cls, args):
        """
        Answer the connection URL for the BOA Server from args.
        :param args: Parsed commandline arguments.
        :return: Connection URL (protocol://hostname:port)
        """
        return f"{args.protocol}://{args.boa_server}:{args.port}"

    @classmethod
    def parse_commandline_args(cls, example_description, default_epochs):
        """
        Parse commmon commandline argument shared by all BOA examples and
        answer the object containing the parsed arguments.
        :param example_description: Description of the example. Used to create the
        usage text for the example.
        :return: Parsed commandline arguments object
        """

        ## Setup argparse for command-line inputs
        argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                            description=example_description)
        argparser.add_argument('--hostname', '-ho',
                               dest='boa_server',
                               action='store',
                               default='localhost',
                               help='Hostname or IP address of BOA server to connect to.')

        argparser.add_argument('--protocol', '-pr',
                               choices=['http', 'https'],
                               dest='protocol',
                               action='store',
                               default='http',
                               help='Protocol to use to connect to BOA server.')

        argparser.add_argument('--port', '-p',
                               type=BoaExamplesUtils.validate_port,
                               dest='port',
                               action='store',
                               default=80,
                               help='Port to connect to on BOA server.')

        argparser.add_argument('--epochs', '-e',
                               type=BoaExamplesUtils.validate_epochs,
                               dest='epochs',
                               action='store',
                               default=default_epochs,
                               help='Number of epochs to perform.')

        argparser.add_argument('--ca-cert-path', '-c',
                               type=BoaExamplesUtils.validate_ca_cert_path,
                               dest='ca_cert_path',
                               action='store',
                               default=None,
                               help='Path to the CA certificate file to use for https protocol.')

        ## Parse command-line arguments
        return argparser.parse_args()

