FROM continuumio/miniconda:4.5.11

# Install JDK.
RUN apt-get update && apt-get install --yes default-jdk

RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
# Setup the Python environment.
RUN conda create --yes --name newsqa python=2.7 "pandas>=0.19.2" cython
RUN echo "conda activate newsqa" >> ~/.bashrc

WORKDIR /usr/src/newsqa
COPY requirements.txt ./
RUN /bin/bash --login -c "conda list && yes | pip install --requirement requirements.txt"

ADD https://nlp.stanford.edu/software/stanford-postagger-2015-12-09.zip /usr/downloads/

# Clean up existing files (there can be problems if they've already been extracted outside of the Docker container).
# Run the unit tests to test and extract the data.
CMD /bin/bash --login -c "rm --force combined-newsqa-data-*.csv maluuba/newsqa/newsqa-data-*.csv && \
                          cp --no-clobber /usr/downloads/* maluuba/newsqa/ && \
                          python -m unittest discover ."
