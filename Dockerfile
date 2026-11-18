FROM python:3.7

WORKDIR /app

COPY requirements-reconstructed.txt /app/

RUN python -m pip install --no-cache-dir \
    -r requirements-reconstructed.txt

COPY . .

CMD ["bash"]
