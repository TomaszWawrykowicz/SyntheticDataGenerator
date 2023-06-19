uci_heart_factorize_params = [('Sex', {'M': 0, 'F': 1}),
                              ('ChestPainType', {'TA': 0, 'ATA': 1, 'NAP': 2, 'ASY': 3}),
                              ('RestingECG', {'Normal': 0, 'ST': 1, 'LVH': 2}),
                              ('ExerciseAngina', {'N': 0, 'Y': 1}),
                              ('ST_Slope', {'Up': 0, 'Flat': 1, 'Down': 2})
                              ]

uci_heart_factorize_back_params = [('Sex', {0: 'M', 1: 'F'}),
                                   ('ChestPainType', {0: 'TA', 1: 'ATA', 2: 'NAP', 3: 'ASY'}),
                                   ('RestingECG', {0: 'Normal', 1: 'ST', 2: 'LVH'}),
                                   ('ExerciseAngina', {0: 'N', 1: 'Y'}),
                                   ('ST_Slope', {0: 'Up', 1: 'Flat', 2: 'Down'})
                                   ]

maternal_factorize_params = [('RiskLevel', {'low risk': 0, 'mid risk': 1, 'high risk': 2})]
maternal_factorize_back_params = [('RiskLevel', {0: 'low risk', 1: 'mid risk', 2: 'high risk'})]


uci_heart_sdv_metadata = {
    "METADATA_SPEC_VERSION": "SINGLE_TABLE_V1",
    "columns": {
        "Age": {
            "sdtype": "numerical"
        },
        "Sex": {
            "sdtype": "categorical"
        },
        "ChestPainType": {
            "sdtype": "categorical"
        },
        "RestingBP": {
            "sdtype": "numerical"
        },
        "Cholesterol": {
            "sdtype": "numerical"
        },
        "FastingBS": {
            "sdtype": "numerical"
        },
        "RestingECG": {
            "sdtype": "categorical"
        },
        "MaxHR": {
            "sdtype": "numerical"
        },
        "ExerciseAngina": {
            "sdtype": "categorical"
        },
        "Oldpeak": {
            "sdtype": "numerical"
        },
        "ST_Slope": {
            "sdtype": "categorical"
        },
        "HeartDisease": {
            "sdtype": "numerical"
        }
    }
}
