ARG TARGETPLATFORM=linux/amd64
FROM --platform=${TARGETPLATFORM} ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Enable universe repo and install Python 3.9, .NET 8 runtime, and DWSIM dependencies
# coinor-libipopt1v5 (required by DWSIM) lives in the universe repo
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository universe \
    && apt-get update && apt-get install -y \
        python3.9 python3.9-dev python3-pip \
        wget ca-certificates gnupg \
        libgdiplus \
        coinor-libipopt1v5 \
    && wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb \
        -O packages-microsoft-prod.deb \
    && dpkg -i packages-microsoft-prod.deb \
    && rm packages-microsoft-prod.deb \
    && apt-get update && apt-get install -y dotnet-runtime-8.0 \
    && rm -rf /var/lib/apt/lists/*

# Copy pre-downloaded DWSIM .deb (avoids unreliable SourceForge download at build time)
# The .deb file is not included in this archive; obtain dwsim_9.0.4-amd64.deb from the
# DWSIM SourceForge release page and place it at docker/dwsim_9.0.4-amd64.deb before building.
COPY docker/dwsim_9.0.4-amd64.deb /tmp/dwsim.deb
RUN apt-get install -y /tmp/dwsim.deb \
    && rm /tmp/dwsim.deb \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY requirements.txt /tmp/requirements.txt
RUN python3.9 -m pip install --upgrade pip && \
    python3.9 -m pip install -r /tmp/requirements.txt
