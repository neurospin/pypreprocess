dataset_names = {
    'ds001': 'Balloon Analog Risk-taking Task',
    'ds002': 'Classification learning',
    'ds003': 'Rhyme judgment',
    'ds005': 'Mixed-gambles task',
    'ds006A': ('Living-nonliving decision with plain '
                 'or mirror-reversed text'),
    'ds007': 'Stop-signal task with spoken & manual responses',
    'ds008': 'Stop-signal task with unselective and selective stopping',
    'ds011': 'Classification learning and tone-counting',
    'ds017A': ('Classification learning and '
                 'stop-signal (1 year test-retest)'),
    'ds017B': ('Classification learning and '
                 'stop-signal (1 year test-retest)'),
    'ds051': 'Cross-language repetition priming',
    'ds052': 'Classification learning and reversal',
    'ds101': 'Simon task dataset',
    'ds102': 'Flanker task (event-related)',
    'ds105': 'Visual object recognition',
    'ds107': 'Word and object processing',
    'ds108': ('Prefrontal-Subcortical Pathways '
                 'Mediating Successful Emotion Regulation'),
    'ds109': 'False belief task',
    'ds110': 'Incidental encoding task (Posner Cueing Paradigm)',
    }

dataset_files = {
    'ds001': ['ds001_raw_6'],
    'ds002': ['ds002_raw_0'],
    'ds003': ['ds003_raw_1'],
    'ds005': ['ds005_raw_0'],
    'ds006A': ['ds006A_raw'],
    'ds007': ['ds007_raw'],
    'ds008': ['ds008_raw_4'],
    'ds011': ['ds011_raw_0'],
    'ds017A': ['ds017A_raw_0'],
    'ds017B': ['ds017B_raw_0'],
    'ds051': ['ds051_raw_0'],
    'ds052': ['ds052_raw_0'],
    'ds101': ['ds101_raw_0'],
    'ds102': ['ds102_raw_0'],
    'ds105': ['ds105_raw_6'],
    'ds107': ['ds107_raw_0'],
    'ds108': ['ds108_raw_part1', 'ds108_raw_part2', 'ds108_raw_part3'],
    'ds109': ['ds109_raw_4'],
    'ds110': ['ds110_raw_part1', 'ds110_raw_part2', 'ds110_raw_part3',
                 'ds110_raw_part4', 'ds110_raw_part5', 'ds110_raw_part6'],
    }


dataset_ignore_list = {
    'ds001': [],
    'ds002': [],
    'ds003': [],
    'ds005': [],
    'ds006A': [],
    'ds007': [
        'sub009',  # missing task002 sessions
        'sub018',  # error on preprocessing
        ],
    'ds008': [
        'sub009',  # missing onsets for task002
        ],
    'ds011': [],
    'ds017A': [
        'sub006',  # missing some onsets
        ],
    'ds017B': [],
    'ds051': [
        'sub006',  # missing run007 & run008
        ],
    'ds052': [],
    'ds101': [],
    'ds102': [],
    'ds105': [],
    'ds107': [],
    'ds108': [],
    'ds109': [],
    'ds110': [],
    }

map_id = {
    'ds001': 'ds000001',
    'ds002': 'ds000002',
    'ds003': 'ds000003',
    'ds005': 'ds000005',
    'ds006A': 'ds000006a',
    'ds007': 'ds000007',
    'ds008': 'ds000008',
    'ds011': 'ds000011',
    'ds017A': 'ds000017',
    'ds017B': 'ds000017',
    'ds051': 'ds000051',
    'ds052': 'ds000052',
    'ds101': 'ds000101',
    'ds102': 'ds000102',
    'ds105': 'ds000105',
    'ds107': 'ds000107',
    'ds108': 'ds000108',
    'ds109': 'ds000109',
    'ds110': 'ds000110'
}
