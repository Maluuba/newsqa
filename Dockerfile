FROM continuumio/miniconda:4.4.10

# Install JDK.
# Inspired by:
# * https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-get-on-debian-8
# * https://github.com/dockerfile/java/blob/master/oracle-java8/Dockerfile
# * https://askubuntu.com/a/15272
RUN echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
    apt-get install --yes software-properties-common gnupg && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C2518248EEA14886 && \
    add-apt-repository --yes "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" && \
    apt-get update && \
    apt-get install --yes --allow-unauthenticated oracle-java8-installer && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/oracle-jdk8-installer

# Setup the Python environment.
RUN conda create --yes --name newsqa python=2.7 "pandas>=0.19.2"
WORKDIR /usr/src/newsqa
COPY requirements.txt ./
RUN /bin/bash -c "source activate newsqa && yes | pip install --requirement requirements.txt"

ADD https://nlp.stanford.edu/software/stanford-postagger-2015-12-09.zip /usr/downloads/

# Clean up existing files (there can be problems if they've already been extracted outside of the Docker container).
# Run the unit tests to test and extract the data.
CMD /bin/bash -c "rm --force combined-newsqa-data-*.csv maluuba/newsqa/newsqa-data-*.csv && \
                  cp --no-clobber /usr/downloads/* maluuba/newsqa/ && \
                  source activate newsqa && \
                  python -m unittest discover ."
