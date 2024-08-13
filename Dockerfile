FROM kubeflownotebookswg/jupyter

ARG PYTORCH_VERSION=2.0.1

# nvidia container toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=12.0"

# install - pytorch (cuda)
RUN python3 -m pip install --quiet --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
    torch==${PYTORCH_VERSION} \
    torchvision
    
USER root

# Needed for fury vtk. ffmpeg also needed
RUN apt-get update \
  && apt-get install ffmpeg libsm6 libxext6 xvfb sudo -y \
  && apt autoremove -y && apt clean -y  \
  && sed -i /etc/sudoers -re 's/^%sudo.*/%sudo ALL=(ALL:ALL) NOPASSWD: ALL/g' && \
    sed -i /etc/sudoers -re 's/^root.*/root ALL=(ALL:ALL) NOPASSWD: ALL/g' && \
    sed -i /etc/sudoers -re 's/^#includedir.*/## **Removed the include directive** ##"/g' && \
    echo "${NB_USER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && \  
    chmod g+w /etc/passwd

COPY --chown=$NB_UID . /app
ENV TOTALSEG_HOME_DIR=/app/.totalsegmentator

RUN pip install --upgrade pip \
  && pip install fury \
  && pip install /app

USER $NB_UID
EXPOSE 8888