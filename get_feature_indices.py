#!/usr/bin/env python3
"""
Script to get feature indices for cat and dog UUIDs
"""
import sys
sys.path.append('src')

from sae_analyzer import SAEAnalyzer

def get_feature_indices():
    analyzer = SAEAnalyzer('meta-llama/Llama-3.3-70B-Instruct')
    
    # Top 5 cat UUIDs from feature discovery
    cat_uuids = [
        'a90b3927-c7dd-4bc6-b372-d8ab8dc8492d',  # Content where cats are the primary subject matter
        '1b026487-ff85-4221-97e0-bb75fe81b323',  # Descriptions of cats lounging and engaging in daily activities  
        '3f007d1b-db22-49bc-898e-3c655d1124a4',  # Portuguese animal words beginning with g, especially gato and gado
        '191ef71a-d4c5-472f-852e-e2a65d53c3c1',  # Living beings or entities under ownership or custody
        '77f2170b-d061-41e0-b2ba-b9ef9f28766e'   # Generalizing statements about cat behaviors and traits
    ]
    
    # Top 5 dog UUIDs from feature discovery
    dog_uuids = [
        '8590febb-885e-46e5-a431-fba0dd1d04af',  # Dogs as loyal and loving companions
        'f7388195-4dce-4cc9-9409-39c1fd1828a7',  # References to dogs as subjects of discussion or description
        'e52d262c-211c-43e4-a97e-c143c3232b40',  # Lists and enumerations of dog breeds
        '84471e05-a043-4e72-a4ed-aa205818a3e2',  # Descriptive text patterns about dog breed personality traits
        'b62ab6c8-f304-4314-a683-18e7478e6b63'   # Narrative content featuring dogs as characters or subjects
    ]
    
    print('🐱 CAT FEATURE INDICES:')
    for i, uuid in enumerate(cat_uuids, 1):
        try:
            feature = analyzer.get_target_feature(uuid)
            print(f'  {i}. Index {feature.index_in_sae}: {feature.label}')
            print(f'     UUID: {uuid}')
        except Exception as e:
            print(f'  {i}. UUID {uuid}: Error - {e}')
        print()
    
    print('\n🐶 DOG FEATURE INDICES:')
    for i, uuid in enumerate(dog_uuids, 1):
        try:
            feature = analyzer.get_target_feature(uuid)
            print(f'  {i}. Index {feature.index_in_sae}: {feature.label}')
            print(f'     UUID: {uuid}')
        except Exception as e:
            print(f'  {i}. UUID {uuid}: Error - {e}')
        print()

if __name__ == "__main__":
    get_feature_indices()