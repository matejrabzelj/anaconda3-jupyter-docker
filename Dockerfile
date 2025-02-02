# Anaconda3 / Miniconda3 Dockerfile
# Copyright (C) 2020-2023  Chelsea E. Manning
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

FROM ubuntu:noble-20241118.1
LABEL description="Anaconda3 Vanilla Container"

# $ docker build --network=host -t matejrabzelj/anaconda3:latest -f Dockerfile .
# $ docker run --rm -it matejrabzelj/anaconda3:latest /bin/bash
# $ docker push matejrabzelj/anaconda3:latest

ARG ANACONDA_CONTAINER="v24.11.3"
ARG ANACONDA_DIST="Miniconda3"
ARG ANACONDA_PYTHON="py312"
ARG ANACONDA_CONDA="24.11.3"
ARG ANACONDA_OS="Linux"
ARG ANACONDA_ARCH="x86_64"
ARG ANACONDA_FLAVOR="Miniforge3"
ARG ANACONDA_PATCH="0"
ARG ANACONDA_VERSION="${ANACONDA_CONDA}-${ANACONDA_PATCH}"
ARG ANACONDA_INSTALLER="${ANACONDA_FLAVOR}-${ANACONDA_VERSION}-${ANACONDA_OS}-${ANACONDA_ARCH}.sh"
ARG ANACONDA_ENV="base"
ARG ANACONDA_GID="100"
ARG ANACONDA_PATH="/usr/local/anaconda3"
ARG ANACONDA_UID="1100"
ARG ANACONDA_USER="anaconda"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

ENV DEBIAN_FRONTEND=noninteractive

# Update repository
RUN apt-get update --fix-missing

# Install dependencies
RUN apt-get install -y --no-install-recommends \
    bzip2 \
    ca-certificates \
    curl \
    locales \
    sudo \
    wget

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

# Configure environment
ENV ANACONDA_ENV=${ANACONDA_ENV} \
    ANACONDA_PATH=${ANACONDA_PATH} \
    ANACONDA_GID=${ANACONDA_GID} \
    ANACONDA_UID=${ANACONDA_UID} \
    ANACONDA_USER=${ANACONDA_USER} \
    HOME=/home/${ANACONDA_USER} \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    SHELL=/bin/bash

ENV PATH=${ANACONDA_PATH}/bin:${PATH}

# Enable prompt color, generally
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

# Copy fix-permissions script
COPY scripts/fix-permissions /usr/local/bin/fix-permissions

# Create default user wtih name "anaconda"
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su \
    && sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers \
    && sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers \
    && useradd -m -s /bin/bash -N -u ${ANACONDA_UID} ${ANACONDA_USER} \
    && mkdir -p ${ANACONDA_PATH} \
    && chown -R ${ANACONDA_USER}:${ANACONDA_GID} ${ANACONDA_PATH} \
    && chmod g+w /etc/passwd \
    && chmod a+rx /usr/local/bin/fix-permissions \
    && fix-permissions ${HOME} \
    && fix-permissions ${ANACONDA_PATH}

# Switch to user "anaconda"
USER ${ANACONDA_UID}
WORKDIR ${HOME}

# Install Anaconda (Miniconda) - https://anaconda.com/
RUN wget --verbose -O ~/${ANACONDA_VERSION}.sh https://github.com/conda-forge/miniforge/releases/download/${ANACONDA_VERSION}/${ANACONDA_INSTALLER} \
    && /bin/bash /home/${ANACONDA_USER}/${ANACONDA_VERSION}.sh -b -u -p ${ANACONDA_PATH} \
    && chown -R ${ANACONDA_USER} ${ANACONDA_PATH} \
    && rm -rvf ~/${ANACONDA_VERSION}.sh \
    && echo ". ${ANACONDA_PATH}/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate \${ANACONDA_ENV}" >> ~/.bashrc \
    && find ${ANACONDA_PATH} -follow -type f -name '*.a' -delete \
    && find ${ANACONDA_PATH} -follow -type f -name '*.js.map' -delete \
    && fix-permissions ${HOME} \
    && fix-permissions ${ANACONDA_PATH}

# Update Anaconda
RUN conda update -c defaults conda

# Activate conda-forge
RUN conda config --add channels conda-forge

# Install Tini
RUN conda install -y tini

# Switch back to root
USER root

# Clean Anaconda
RUN conda clean -afy \
    && fix-permissions ${HOME} \
    && fix-permissions ${ANACONDA_PATH}

# Make configuration adjustments in /etc
RUN ln -s ${ANACONDA_PATH}/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && fix-permissions /etc/profile.d/conda.sh

# Clean packages and caches
RUN apt-get --purge -y remove wget curl \
    && apt-get --purge -y autoremove \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* \
    && rm -rvf /home/${ANACONDA_PATH}/.cache/yarn

# Configure container startup
ENTRYPOINT [ "tini", "-g", "--" ]
CMD [ "/bin/bash" ]

# Re-activate user "anaconda"
USER $ANACONDA_UID
WORKDIR $HOME
