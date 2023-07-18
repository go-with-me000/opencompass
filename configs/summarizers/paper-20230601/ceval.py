from mmengine.config import read_base

with read_base():
    from ..groups.ceval import ceval_summary_groups

summarizer = dict(
    dataset_abbrs = [
        # CEval
        'ceval-advanced_mathematics',
        'ceval-college_chemistry',
        'ceval-college_physics',
        'ceval-college_programming',
        'ceval-computer_architecture',
        'ceval-computer_network',
        'ceval-discrete_mathematics',
        'ceval-electrical_engineer',
        'ceval-high_school_biology',
        'ceval-high_school_chemistry',
        'ceval-high_school_mathematics',
        'ceval-high_school_physics',
        'ceval-metrology_engineer',
        'ceval-middle_school_biology',
        'ceval-middle_school_chemistry',
        'ceval-middle_school_mathematics',
        'ceval-middle_school_physics',
        'ceval-operating_system',
        'ceval-probability_and_statistics',
        'ceval-veterinary_medicine',
        'ceval-business_administration',
        'ceval-college_economics',
        'ceval-education_science',
        'ceval-high_school_geography',
        'ceval-high_school_politics',
        'ceval-mao_zedong_thought',
        'ceval-marxism',
        'ceval-middle_school_geography',
        'ceval-middle_school_politics',
        'ceval-teacher_qualification',
        'ceval-art_studies',
        'ceval-chinese_language_and_literature',
        'ceval-high_school_chinese',
        'ceval-high_school_history',
        'ceval-ideological_and_moral_cultivation',
        'ceval-law',
        'ceval-legal_professional',
        'ceval-logic',
        'ceval-middle_school_history',
        'ceval-modern_chinese_history',
        'ceval-professional_tour_guide',
        'ceval-accountant',
        'ceval-basic_medicine',
        'ceval-civil_servant',
        'ceval-clinical_medicine',
        'ceval-environmental_impact_assessment_engineer',
        'ceval-fire_engineer',
        'ceval-physician',
        'ceval-plant_protection',
        'ceval-sports_science',
        'ceval-tax_accountant',
        'ceval-urban_and_rural_planner',
        "ceval-stem",
        "ceval-social-science",
        "ceval-humanities",
        "ceval-other",
        "ceval",
        "ceval-hard",
    ],
    summary_groups=ceval_summary_groups,
    prompt_db=dict(
        database_path='configs/datasets/log.json',
        config_dir='configs/datasets',
        blacklist='.promptignore')
)
