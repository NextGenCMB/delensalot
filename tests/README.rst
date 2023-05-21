Tests
======

List of useful tests:


integration
------------

 - delensalot models
    - Check if custom model is build
    - Check if default values are loaded
    - Check if ill-defined model is aborted
 - delensalot job inits (this is model build and init),
    - check if all attributes (input and built at runtime) match expectation
    - check if all job methods do as expected
 - delensalot job runs
    - Check if all products are generated
 - dependencies
    - plancklens
    - lenspyx
    - ducc  

unit
-----

 - delensalot model attributes
    - Check if attribute does as intended
 - utilities
    - Check if methods do as expected
 - conventions
    - TBD, not sure how to test
 - module methods
    - CHeck if filter, truncation, prediction, response, etc. work as expected?