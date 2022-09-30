from rest_framework import viewsets
from rest_framework import mixins

from apps.endpoints.models import Measurements
from apps.endpoints.serializers import MeasurementSerializer

from apps.endpoints.models import Endpoint
from apps.endpoints.serializers import EndpointSerializer

# from apps.endpoints.models import MLAlgorithm
# from apps.endpoints.serializers import MLAlgorithmSerializer

# from apps.endpoints.models import MLAlgorithmStatus
# from apps.endpoints.serializers import MLAlgorithmStatusSerializer

# from apps.endpoints.models import MLRequest
# from apps.endpoints.serializers import MLRequestSerializer

# from django.db import transaction
# from apps.endpoints.models import ABTest
# from apps.endpoints.serializers import ABTestSerializer

import json
import pandas as pd
import time
from numpy.random import rand
from rest_framework import views, status
from rest_framework.response import Response
# from apps.ml.registry import MLRegistry
from apps.ml.income_classifier.measurement import Measurement, NotEnoughData, TimeStampTooHigh, SynchronizationError
from server.wsgi import mlp
from decimal import Decimal
import traceback


# from server.wsgi import registry


class MeasurementViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = MeasurementSerializer
    queryset = Measurements.objects.all()


class EndpointViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


# class MLAlgorithmViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet):
#     serializer_class = MLAlgorithmSerializer
#     queryset = MLAlgorithm.objects.all()


# def deactivate_other_statuses(instance):
#     old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm=instance.parent_mlalgorithm,
#                                                     created_at__lt=instance.created_at,
#                                                     active=True)
#     for i in range(len(old_statuses)):
#         old_statuses[i].active = False
#     MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])


# class MLAlgorithmStatusViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
#                                mixins.CreateModelMixin):
#     serializer_class = MLAlgorithmStatusSerializer
#     queryset = MLAlgorithmStatus.objects.all()
#
#     def perform_create(self, serializer):
#         try:
#             with transaction.atomic():
#                 instance = serializer.save(active=True)
#                 # set active=False for other statuses
#                 deactivate_other_statuses(instance)
#
#         except Exception as e:
#             raise APIException(str(e))
#
#
# class MLRequestViewSet(mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
#                        mixins.UpdateModelMixin):
#     serializer_class = MLRequestSerializer
#     queryset = MLRequest.objects.all()


# class PredictView(views.APIView):
#     def post(self, request, endpoint_name, format=None):
#
#         algorithm_status = self.request.query_params.get("status", "production")
#         algorithm_version = self.request.query_params.get("version")
#
#         algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name, status__status=algorithm_status,
#                                           status__active=True)
#
#         if algorithm_version is not None:
#             algs = algs.filter(version=algorithm_version)
#
#         if len(algs) == 0:
#             return Response(
#                 {"status": "Error", "message": "ML algorithm is not available"},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )
#         if len(algs) != 1 and algorithm_status != "ab_testing":
#             return Response(
#                 {"status": "Error",
#                  "message": "ML algorithm selection is ambiguous. Please specify algorithm version."},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )
#         alg_index = 0
#         if algorithm_status == "ab_testing":
#             alg_index = 0 if rand() < 0.5 else 1
#
#         algorithm_object = registry.endpoints[algs[alg_index].id]
#         prediction = algorithm_object.compute_prediction(request.data)
#
#         label = prediction["label"] if "label" in prediction else "error"
#         ml_request = MLRequest(
#             input_data=json.dumps(request.data),
#             full_response=prediction,
#             response=label,
#             feedback="",
#             parent_mlalgorithm=algs[alg_index],
#         )
#         ml_request.save()
#
#         prediction["request_id"] = ml_request.id
#
#         return Response(prediction)


class SaveAndPredictView(views.APIView):
    """
        {"measure": [{"limp": "f",
                     "side": "l",
                     "type": "g",
                     "values": [{"timestamp": 1224L,  # 1639427740008
                                 "vector": {"v1": 123f,
                                            "v2": 1234f,
                                            "v3": 322f
                                           }
                                },
                                {"timestamp": 1224L,
                                 "vector": {"v1": 123f,
                                            "v2": 1234f,
                                            "v3": 322f
                                           }
                                }...
                               ]
                    },
                    {"limp": "a",
                     "side": "r",
                     "type": "a",
                     "values": [{"timestamp": 1224L,
                                 "vector": {"v1": 123f,
                                            "v2": 1234f,
                                            "v3": 322f
                                           }
                                },
                                {"timestamp": 1224L,
                                 "vector": {"v1": 123f,
                                            "v2": 1234f,
                                            "v3": 322f
                                           }
                                }...
                                ]
                    }
                   ]...
        }
    """

    def post(self, request, endpoint_name, format=None):
        def load_ml_algorithm():
            return mlp

        # def old_load_ml_algorithm():
        #     algorithm_status = self.request.query_params.get("status", "production")
        #     algorithm_version = "0.0.1"  # self.request.query_params.get("version")
        #
        #     algs = MLAlgorithm.objects.filter(parent_endpoint__name=endpoint_name, status__status=algorithm_status,
        #                                       status__active=True)
        #
        #     if algorithm_version is not None:
        #         algs = algs.filter(version=algorithm_version)
        #
        #     # TODO: we have more algs from the same type, but the need the first now
        #     # if len(algs) == 0:
        #     #     raise ValueError("status: Error, message: ML algorithm is not available")
        #     # if len(algs) != 1 and algorithm_status != "ab_testing":
        #     #     raise ValueError("status: Error, message: ML algorithm selection is ambiguous."
        #     #                      " Please specify algorithm version.")
        #
        #     # TODO
        #     # alg_index = 0
        #     # _algorithm_object = registry.endpoints[algs[alg_index].id]
        #     _algorithm_object = registry.endpoints[3]
        #     return _algorithm_object

        def write_data_into_db(chunk_size):
            if isinstance(request.data, dict):
                _input_data = json.loads(json.dumps(request.data))
            elif isinstance(request.data, str):
                _input_data = json.loads(request.data)
            else:
                raise TypeError("request.data is not dict or str")

            measurement_id = _input_data["measurement_id"]
            measurement_list = list()
            first_timestamp = float("inf")
            for measurement_item in _input_data["measure"]:
                for value in measurement_item["values"]:
                    if value["timestamp"] < first_timestamp:
                        first_timestamp = value["timestamp"]

                    measurement = Measurements(
                        measurement_id=measurement_id,
                        timestamp=value["timestamp"],
                        measurement_type=measurement_item["type"],
                        limp_type=measurement_item["limp"],
                        limp_side=measurement_item["side"],
                        v1=Decimal(str(round(value["vector"]["v1"], 3))),
                        v2=Decimal(str(round(value["vector"]["v2"], 3))),
                        v3=Decimal(str(round(value["vector"]["v3"], 3))),
                    )
                    measurement_list.append(measurement)

                    if len(measurement_list) == chunk_size:
                        Measurements.objects.bulk_create(measurement_list)
                        measurement_list = list()

            if len(measurement_list) > 0:
                Measurements.objects.bulk_create(measurement_list)

            assert first_timestamp < float("inf")
            return _input_data, first_timestamp

        def get_meas_from_db(meas_length_min, first_timestamp_ms):
            key_list = [("left", "arm", "acc"),
                        ("left", "arm", "gyr"),
                        ("left", "leg", "acc"),
                        ("left", "leg", "gyr"),
                        ("right", "arm", "acc"),
                        ("right", "arm", "gyr"),
                        ("right", "leg", "acc"),
                        ("right", "leg", "gyr")]

            key_map = {
                ("left", "arm", "acc"):  ("l", "a", "a"),
                ("left", "arm", "gyr"):  ("l", "a", "g"),
                ("left", "leg", "acc"):  ("l", "f", "a"),
                ("left", "leg", "gyr"):  ("l", "f", "g"),
                ("right", "arm", "acc"): ("r", "a", "a"),
                ("right", "arm", "gyr"): ("r", "a", "g"),
                ("right", "leg", "acc"): ("r", "f", "a"),
                ("right", "leg", "gyr"): ("r", "f", "g"),
            }

            meas_length_ms = int(meas_length_min * 60 * 1e3)
            # current_timestamp_ms = time.time_ns() * 1e6

            start_timestamp = first_timestamp_ms - meas_length_ms
            measurement_id = input_data["measurement_id"]

            def make_query(_keys):
                query = Measurements.objects.filter(timestamp__gte=start_timestamp,
                                                    measurement_id=measurement_id,
                                                    measurement_type=key_map[_keys][2],
                                                    limp_type=key_map[_keys][1],
                                                    limp_side=key_map[_keys][0]).values()
                return query

            def df_from_query(queries):
                df_dict = {'timestamp': list(),
                           'v1': list(),
                           'v2': list(),
                           'v3': list()}

                cast_dict = {'timestamp': lambda x: x,
                             'v1': lambda x: float(x),
                             'v2': lambda x: float(x),
                             'v3': lambda x: float(x)}

                for query in queries:
                    for key in df_dict.keys():
                        df_dict[key].append(cast_dict[key](query[key]))

                return pd.DataFrame.from_dict(df_dict)

            _meas = Measurement(measurement_id)
            _meas.measurement_dict = {keys: df_from_query(make_query(keys)) for keys in key_list}
            return _meas

        def get_instances(meas_length_min, inference_delta_sec, first_timestamp_ms):
            frequency = 25  # Hz, T = 40 ms
            expected_delta = (1 / frequency) * 1000  # ms
            eps = 3
            meas.check_frequency(expected_delta, eps=eps)
            meas.synchronize_measurement_dict()

            length = frequency * 60 * meas_length_min
            inference_delta_ms = inference_delta_sec * 1e3
            keys_in_order = (("arm", "acc"),
                             ("leg", "acc"),
                             ("arm", "gyr"),
                             ("leg", "gyr"))

            def collect_instances():
                _instance_list = list()
                _inference_ts_list = list()
                i = 0
                while True:
                    end_ts = int(first_timestamp_ms + inference_delta_ms * i)
                    i += 1
                    _instance = list()
                    for key in keys_in_order:
                        try:
                            diff_mean = meas.get_limb_diff_mean(key[0], key[1], length, end_ts=end_ts)
                            ratio_mean_first = meas.get_limb_ratio_mean(key[0], key[1], length, end_ts=end_ts,
                                                                        mean_first=True)
                            ratio_mean = meas.get_limb_ratio_mean(key[0], key[1], length, end_ts=end_ts,
                                                                  mean_first=False)
                        except NotEnoughData:
                            break
                        except TimeStampTooHigh:
                            return _instance_list, _inference_ts_list

                        _instance.append([diff_mean, ratio_mean_first, ratio_mean])

                    if len(_instance) > 0:
                        _instance_list.append(sum(_instance, []))
                        _inference_ts_list.append(end_ts)

            return collect_instances()

        try:
            _chunk_size = 10000
            input_data, _first_timestamp_ms = write_data_into_db(_chunk_size)
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return Response({"status": "Error during write data into the db: {}, "
                                       "{}".format(repr(e), traceback.format_exc()),
                             "prediction": "None"})

        try:
            _meas_length_min = 90
            meas = get_meas_from_db(_meas_length_min, _first_timestamp_ms)
        except Exception as e:
            print(e)
            return Response({"status": "Error during get measurement: {}".format(repr(e)), "prediction": "None"})

        try:
            _inference_delta_sec = 30  # sec
            instances, inference_ts_list = get_instances(_meas_length_min, _inference_delta_sec, _first_timestamp_ms)
            if len(instances) == 0:
                print("No prediction, more data needed")
                return Response({"status": "No prediction, more data needed", "prediction": "None"})
        except SynchronizationError:
            print("No prediction, more data needed or there is problem with the frequency")
            return Response({"status": "No prediction, more data needed or there is problem with the frequency",
                             "prediction": "None"})
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            return Response({"status": "Error during get instance: {}".format(repr(e)), "prediction": "None"})

        try:
            algorithm_object = load_ml_algorithm()
        except Exception as e:
            print(e)
            return Response({"status": "Error during load the algorithm: {}".format(repr(e)), "prediction": "None"})

        try:
            prediction = algorithm_object.compute_prediction(instances, inference_ts_list)

        except Exception as e:
            print(e)
            return Response({"status": "Error during make prediction: {}".format(repr(e)), "prediction": "None"})

        print(prediction)
        return Response({"status": "OK", "prediction": prediction})
