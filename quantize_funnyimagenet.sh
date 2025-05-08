#!/bin/sh
python quantize.py funnyimagenet/qat_best.pth.tar trained/ai85-funnyimagenet-qat8-q.pth.tar --device MAX78000 -v "$@"
