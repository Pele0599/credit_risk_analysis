from operon.sklearn import SymbolicRegressor

est = SymbolicRegressor(
            local_iterations=5,
            generations=10000, 
            n_threads=1,
            random_state=None,
            time_limit=1*60*60, # 1 hour
            max_evaluations=int(5e5),
            population_size=500
            )

hyper_params = {
    'population_size': (100,),
    'pool_size': (100,),
    'max_length': (25,),
    'allowed_symbols': ('add,mul,aq,constant,variable','exp'),
    'local_iterations': (5,),
    'offspring_generator': ('basic',),
    'tournament_size': (3,),
    'reinserter': ('keep-best',),
    'max_evaluations': (int(5e5),),
    'random_state' : 42, 
}

cv = KFold(n_splits=1, shuffle=True,random_state=42)

grid_est = HalvingGridSearchCV(est,cv=cv, param_grid=hyper_params,
        verbose=2, n_jobs=1, scoring='r2', error_score=0.0)
    
