.. _sec_av_s3:

=========
S3 access
=========

Since DC datasets can become quite large, it often makes sense to put them
somewhere centrally, such as a shared network drive or :ref:`DCOR <sec_av_dcor>`.
You may also choose to upload your files directly to an
`S3-compatible object store <https://en.wikipedia.org/wiki/Amazon_S3>`_, which
dclab supports as well (It is actually in integral part of the DCOR format).

Public data
===========

Opening public datasets on S3 is straight forward. To get started, you only
need to know the URL of the object:

.. code:: python

    import dclab
    s3_url = "https://objectstore.hpccloud.mpcdf.mpg.de/circle-5a7a053d-55fb-4f99-960c-f478d0bd418f/resource/fb7/19f/b2-bd9f-817a-7d70-f4002af916f0"
    ds = dclab.new_dataset(s3_url)
    print(ds.config)


.. _sec_av_s3_private:

Private data
============

Accessing private data requires you to pass the key ID and the
access secret like so:

.. code:: python

    import dclab
    s3_url = "..."
    ds = dclab.new_dataset(s3_url,
                           secret_id="YOUR-S3-KEY-ID",
                           secret_key="YOUR-S3-ACCESS-SECRET")


Alternatively, you can also set the environment variables
``DCLAB_S3_ACCESS_KEY_ID`` and ``DCLAB_S3_SECRET_ACCESS_KEY``, and
optionally the ``DCLAB_S3_ENDPOINT_URL``. If you cannot edit environment
variables, you can also modify the environment before importing dclab like
so:

.. code:: python

    import os
    os.environ["DCLAB_S3_ACCESS_KEY_ID"] = "4f4bf368365967466be9baf07028a5f3"
    os.environ["DCLAB_S3_ACCESS_KEY_ID"] = "12cd2fe004bc0f17fe9cd76dae412e0d"
    os.environ["DCLAB_S3_ENDPOINT_URL"] = "https://objectstore.hpccloud.mpcdf.mpg.de"

    import dclab
    dclab.new_dataset(...)