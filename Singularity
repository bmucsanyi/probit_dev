Bootstrap: docker
From: python:3.12

%post
    apt install -y libmagickwand-dev

    cd /mnt/path/to/probit
    python -m pip install --root-user-action=ignore -r requirements.txt
    # Uncomment the following lines if you are running on an x86 machine
    # python -m pip uninstall -y pillow
    # CC="cc -mavx2" python -m pip install -U --force-reinstall pillow-simd