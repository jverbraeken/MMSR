# Recipe search and recommendation system

This project is part of the Multimedia Search & Recommendation Course CS4065. The task of this system is to provide a top list of recipes based on user preferences and supermarket deals. For this there are 3 subsystems.
- Subsystem 1 (system1.py) is the search system that looks in a dataset of 20,000 recipes (epi_r.csv and full_format_recipes.json) for the most affordable recipes based on supermarket offers (offers.json)
- Subsystem 2 finds the most favourable recipes for a user based on his previously liked recipes (system2.py)
- Subsystem 3 finds the most favourable recipes based on users with similar taste (system3.py). For this we use a user-item matrix (ratings.csv). 

## How to run the code 
Every single subsystem correponds to one file called system1/2/3.py . These files can be run separately.

The main pipeline that runs all of these systems together and combines the results can be found in pipeline.py . On top of the main function are several parameters that can be adjusted to tune the predictions.