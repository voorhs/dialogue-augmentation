import os
import json
from datasets import load_dataset, DatasetDict, Dataset, disable_caching
from mylib.utils.training import seed_everything


def main(output_path, one_domain, seed):
    seed_everything(seed)
    disable_caching()

    for name in ['MultiWOZ_2.1', 'BiTOD', 'SGD']:
        dataset = DatasetDict({
            split: Dataset.from_generator(
                get_record_generator,
                gen_kwargs=dict(split=split, name=name, one_domain=one_domain),
                cache_dir=False
            ) for split in ['train', 'validation', 'test']
        })

        dataset.save_to_disk(os.path.join(output_path, name))


def get_record_generator(split, name, one_domain):
    dataset = load_dataset('Salesforce/dialogstudio', name=name, split=split, trust_remote_code=True)
    for i, sample in enumerate(dataset):
        uts, sps = [], []
        for turn in sample['log']:
            for sp, item in zip([0, 1], ['user utterance', 'system response']):
                ut = turn[item]
                uts.append(ut)
                sps.append(sp)
        labels = get_domain_labels(name, sample)
        if one_domain and sum(labels) != 1:
            continue
        yield dict(
            content=[{'utterance': ut, 'speaker': sp} for ut, sp in zip(uts, sps)],
            source_dataset_name=f'{name}/{split}',
            idx_within_source=i,
            services=labels
        )


def get_domain_labels(name, sample):
    if name == 'MultiWOZ_2.1':
        return multiwoz_labels(sample)
    if name == 'BiTOD':
        return bitod_labels(sample)
    if name == 'SGD':
        return sgd_labels(sample)


def multiwoz_labels(sample):
    info = json.loads(sample['original dialog info'])['goal']
    labels = [int(len(info[k]) > 0) for k in domains['MultiWOZ_2.1']]
    return labels


def bitod_labels(sample):
    info = json.loads(sample['original dialog info'])['Scenario']
    keys = ''.join(list(info['User_Goal'].keys()))
    labels = [int(k in keys) for k in domains['BiTOD']]
    return labels


def sgd_labels(sample):
    info = json.loads(sample['original dialog info'])['services']
    info = [name[:-2] for name in info]
    labels = [int(k in info) for k in domains['SGD']]
    return labels


domains = {
    'MultiWOZ_2.1': ['attraction', 'hospital', 'hotel', 'restaurant', 'taxi', 'train'],
    'BiTOD': ['attraction', 'hotel', 'restaurant'],
    'SGD': [
        'Banks',
        'Buses',
        'Calendar',
        'Events',
        'Flights',
        'Homes',
        'Hotels',
        'Media',
        'Movies',
        'Music',
        'RentalCars',
        'Restaurants',
        'RideSharing',
        'Services',
        # 'Travel',
        # 'Weather'
    ]
}
