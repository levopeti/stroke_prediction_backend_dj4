from rest_framework import serializers
from apps.endpoints.models import Endpoint, Measurements  # , MLAlgorithm, MLAlgorithmStatus, MLRequest, ABTest,


class MeasurementSerializer(serializers.ModelSerializer):
    class Meta:
        model = Measurements
        read_only_fields = ("id", "timestamp", "measurement_type", "limp_type",
                            "limp_side", "v1", "v2", "v3", "created_at")
        fields = read_only_fields


class EndpointSerializer(serializers.ModelSerializer):
    class Meta:
        model = Endpoint
        read_only_fields = ("id", "name", "owner", "created_at")
        fields = read_only_fields


# class MLAlgorithmSerializer(serializers.ModelSerializer):
#     current_status = serializers.SerializerMethodField(read_only=True)
#
#     def get_current_status(self, mlalgorithm):
#         return MLAlgorithmStatus.objects.filter(parent_mlalgorithm=mlalgorithm).latest('created_at').status
#
#     class Meta:
#         model = MLAlgorithm
#         read_only_fields = ("id", "name", "description", "code",
#                             "version", "owner", "created_at",
#                             "parent_endpoint", "current_status")
#         fields = read_only_fields
#
#
# class MLAlgorithmStatusSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = MLAlgorithmStatus
#         read_only_fields = ("id", "active")
#         fields = ("id", "active", "status", "created_by", "created_at",
#                   "parent_mlalgorithm")
#
#
# class MLRequestSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = MLRequest
#         read_only_fields = (
#             "id",
#             "input_data",
#             "full_response",
#             "response",
#             "created_at",
#             "parent_mlalgorithm",
#         )
#         fields = (
#             "id",
#             "input_data",
#             "full_response",
#             "response",
#             "feedback",
#             "created_at",
#             "parent_mlalgorithm",
#         )
#
#
# class ABTestSerializer(serializers.ModelSerializer):
#     class Meta:
#         model = ABTest
#         read_only_fields = (
#             "id",
#             "ended_at",
#             "created_at",
#             "summary",
#         )
#         fields = (
#             "id",
#             "title",
#             "created_by",
#             "created_at",
#             "ended_at",
#             "summary",
#             "parent_mlalgorithm_1",
#             "parent_mlalgorithm_2",
#         )
