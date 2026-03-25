---
title: Retention Rate Approximator
sdk: gradio
app_file: app.py
---

# Retention-rate-approximator

This project approximates retention rate with three components: the main trend function, patch offsets, and weekly seasonality.

The original notebook remains the source of truth for the research workflow. The repository now also contains a typed Python core and a `Gradio` app for Hugging Face Spaces in `app.py`.

The Hugging Face Space is intended as the easiest way to try the model without opening the notebook locally.

The Space expects CSV files with columns:
- `date`
- `installs`
- `retention`
- `retention_mean`

You can find all the details in [my article](https://habr.com/ru/articles/732882/).

Dmitry Baltin, 2023
