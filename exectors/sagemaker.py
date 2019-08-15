# Python libraries
import os
from typing import Dict, Optional

# Amazon libraries
from boto3.session import Session as BotoSession
from sagemaker.session import Session


class Sagemaker:
    """
    Class to provide AWS specific execution of the models.
    In the future, we can make a superclass that defines the basic methods (such as
    uploading data to the right folder/location, loading models etc).
    For now, we will only have AWS.
    This will be very similar to default session objects.
    """

    training_instance_count = 1
    training_instance_type = "ml.m4.xlarge"
    transformer_instance_count = 1
    transformer_instance_type = "ml.c4.xlarge"
    deploy_instance_count = 1
    deploy_instance_type = "ml.c4.xlarge"

    def __init__(
        self,
        bucket: Optional[str] = None,
        role: Optional[str] = None,
        prefix: Optional[str] = None,
        default_model_kwargs: Optional[Dict] = None,
        default_transfomer_kwargs: Optional[Dict] = None,
        default_deploy_kwargs: Optional[Dict] = None,
    ) -> None:
        """
        Initializes the AWS object

        Arguments:
            bucket: The bucket name. Defaulted to the session default bucket
            role: The role name to assume. Default is getting from AWS_DEFAULT_ROLE of the env variables
            prefix: The prefix to use in the bucket. Defaulted to 'data'
            default_model_kwargs: Dict for default kwargs for any sagemaker model. 
                Default contains train_instance_type, train_instance_count, role and session
            default_transformer_kwargs: Dict for default kwargs for any sagemaker transformer. 
                Default contains instance_type, instance_count, and role.
            default_deploy_kwargs: Dict for default kwargs for any sagemaker deployment. 
                Default contains instance_type and initial_instance_count.
        """
        self.boto_session = BotoSession(
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )
        self.region = self.boto_session.region_name
        self.session = Session(boto_session=self.boto_session)
        self.role = role if role is not None else os.environ.get("AWS_DEFAULT_ROLE")
        self.bucket = bucket if bucket is not None else self.session.default_bucket()
        self.prefix = prefix if prefix is not None else "data"
        self.default_model_kwargs = self._default_model_kwargs(
            self.role, self.session, default_model_kwargs
        )
        self.default_transformer_kwargs = self._default_transformer_kwargs(
            self.role, self.session, default_transfomer_kwargs
        )
        self.default_deploy_kwargs = self._default_deploy_kwargs(
            self.role, self.session, default_deploy_kwargs
        )

    def _default_model_kwargs(self, role, session, input_default) -> Dict:
        initial = {
            "role": role,
            "sagemaker_session": session,
            "train_instance_count": self.training_instance_count,
            "train_instance_type": self.training_instance_type,
        }
        if input_default is not None:
            initial.update(input_default)
        return initial

    def _default_transformer_kwargs(self, role, session, input_default) -> Dict:
        initial = {
            "role": role,
            "instance_count": self.transformer_instance_count,
            "instance_type": self.transformer_instance_type,
        }
        if input_default is not None:
            initial.update(input_default)
        return initial

    def _default_deploy_kwargs(self, role, session, input_default) -> Dict:
        initial = {
            "initial_instance_count": self.deploy_instance_count,
            "instance_type": self.deploy_instance_type,
        }
        if input_default is not None:
            initial.update(input_default)
        return initial

    def upload_data(
        self,
        local_data_file: str,
        bucket: Optional[str] = None,
        prefix: Optional[str] = None,
    ) -> str:
        """
        Uploads the data from the local data file to S3. Returns the location

        Argument:
            local_data_file: the location of the data
            bucket: The bucket to upload to. Defaulted to the own default bucket
            prefix: The prefix to use to upload to. Defaulted to the own default bucket

        Returns:
            The s3 data location
        """
        if bucket is None:
            bucket = self.bucket
        if prefix is None:
            prefix = self.prefix
        return self.session.upload_data(
            local_data_file, bucket=bucket, key_prefix=prefix
        )
