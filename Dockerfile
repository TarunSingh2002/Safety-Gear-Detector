FROM public.ecr.aws/lambda/python:3.9

# Install OS-level dependencies
RUN yum install -y \
    mesa-libGL \
    mesa-libGLU \
    libXext \
    libXrender \
    libSM \
    ffmpeg \
    && yum clean all

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app.py ./
COPY templates ./templates
COPY static ./static
COPY models ./models

CMD ["app.handler"]
