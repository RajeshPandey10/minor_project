FROM python:3.11-slim

# Install system dependencies and Google Chrome
RUN apt-get update && \
    apt-get install -y wget gnupg2 curl && \
    wget -q -O - https://dl.google.com/linux/linux_signing_key.pub | apt-key add - && \
    echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list && \
    apt-get update && \
    apt-get install -y google-chrome-stable && \
    ln -s /usr/bin/google-chrome-stable /usr/bin/google-chrome && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
# Copy and install Python dependencies. Use './' to refer to the current directory.
COPY ./requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire API code into the container
COPY . /app/
CMD ["python", "server1.py"]
