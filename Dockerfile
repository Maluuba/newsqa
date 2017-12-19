FROM continuumio/miniconda

# Setup the Python environment.
RUN conda create --yes --name newsqa python=2.7 "pandas>=0.19.2" 
COPY requirements.txt ./
RUN /bin/bash -c "source activate newsqa && type python && yes | pip install --requirement requirements.txt"

# TODO Download required files.

# TODO Install JDK.

WORKDIR /usr/src/newsqa
# Running the unit tests extracts the data.
CMD /bin/bash -c "source activate newsqa && python -m unittest discover ."
