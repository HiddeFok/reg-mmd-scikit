.. _example1:

Example 1: Estimation
======================

Here is an example of how to use the ``regmmd.estimation`` module:

.. code-block:: python

   from regmmd import MMDEstimator

   # Example code here
   model = MMDEstimator()
   model.fit(X, y)


.. literalinclude:: ../../examples/estimation/gaussian.py
    :language: python
    :linenos:
    :caption: Another estimation example.
