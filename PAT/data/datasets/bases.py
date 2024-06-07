import copy
import logging
import os
import numpy as np
import random


class Dataset(object):
    """An abstract class representing a Dataset.
    This is the base class for ``ImageDataset`` and ``VideoDataset``.
    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
        
    """
    _junk_pids = []  # contains useless person IDs, e.g. background, false detections

    def __init__(self, train, query, gallery, transform=None, mode='train',
                 combineall=False, verbose=True, **kwargs):
        
        if kwargs.get('pose_exp', False):
            train = self.modify_train_with_pose(train)
        self.train = train
        self.query = query
        self.gallery = gallery
        self.query = [tuple(q_tuple)+({'q_or_g': 'query'},) for q_tuple in self.query]
        self.gallery = [tuple(g_tuple)+({'q_or_g': 'gallery'},) for g_tuple in self.gallery]
        self.transform = transform
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose
        
        # half and half experiment
        TEST_0605 = True
        if TEST_0605:
            self.query, self.gallery = self.modify_query_and_gallery(self.query, self.gallery)

        # if self.train != []:
        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)

        if self.combineall:
            self.combine_all()

        if self.mode == 'train':
            self.data = self.train
        elif self.mode == 'query':
            self.data = self.query
        elif self.mode == 'gallery':
            self.data = self.gallery
        else:
            raise ValueError('Invalid mode. Got {}, but expected to be '
                             'one of [train | query | gallery]'.format(self.mode))

        # if self.verbose:
        #     self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def modify_query_and_gallery(self, query, gallery, seed = 123123):
        """
        eg) 1~ 10 pid
        half of the query -> front (1~5)
        half of the query -> back (6~10)
        half of the gallery -> back (1~5)
        half of the gallery -> front (6~10)
        """
        random.seed(seed)
        dataset_list = ["market1501", "MSMT17_V2", "cuhk03"]
        for d in dataset_list:
            if d in query[0][0]:
                dataset_name = d
                break
        dataset_path = os.path.join(query[0][0].split(dataset_name)[0], dataset_name)
        pose_path = os.path.join(dataset_path, 'posture.txt')
        pose_dict = {}
        with open(pose_path, 'r') as f:
            for line in f:
                pose_dict[line.split(' ')[0]] = int(line.split(' ')[1])
        
        query = sorted(query, key=lambda x: int(x[1]))
        query = np.array(query)
        query_pid_array = np.array([int(x[1]) for x in query])
        gallery = sorted(gallery, key=lambda x: int(x[1]))
        gallery = np.array(gallery)
        gallery_pid_array = np.array([int(x[1]) for x in gallery])
        
        pid_set_query = set()
        pid_set_gallery = set()
        for q in query:
            pid_set_query.add(q[1])
        for g in gallery:
            pid_set_gallery.add(g[1])
            
        all_pid = pid_set_query.union(pid_set_gallery)
        
        # examine all pid in query is in gallery
        not_in_gallery = []
        not_in_query = []
        for q in pid_set_query:
            if q not in pid_set_gallery:
                not_in_gallery.append(q)
        for g in pid_set_gallery:
            if g not in pid_set_query:
                not_in_query.append(g)
        print("not in query pids: ", not_in_query, "total pid num in query: ", len(pid_set_query))
        if len(not_in_gallery) > 0:
            raise ValueError("not all query pids are in gallery")
        
        # modify query, gallery
        new_query = []
        new_gallery = []
        passed_pid = []
        front_in_query = []
        back_in_query = []
        free_pid = []
        for pid in sorted(list(all_pid)):
            pid_query = query[query_pid_array == pid]
            pid_gallery = gallery[gallery_pid_array == pid]

            if (pid not in pid_set_query) and (pid in pid_set_gallery):
                # abandon the pid
                passed_pid.append(pid)
                
            else:
                pose_dirs_q = [pose_dict[x[0].split(dataset_name)[1][1:]] for x in pid_query]
                pose_dirs_q = np.array(pose_dirs_q)
                front_query = np.where(pose_dirs_q == 1)[0]
                back_query = np.where(pose_dirs_q == -1)[0]
                
                pose_dirs_g = [pose_dict[x[0].split(dataset_name)[1][1:]] for x in pid_gallery]
                pose_dirs_g = np.array(pose_dirs_g)
                front_gallery = np.where(pose_dirs_g == 1)[0]
                back_gallery = np.where(pose_dirs_g == -1)[0]
                
                if (len(front_query) > 0 and len(back_gallery) > 0):
                    if (len(front_gallery) > 0 and len(back_query) > 0):
                        free_pid.append(pid)
                    else:
                        # random sample the front one from query
                        rand_idx = random.choice(front_query)
                        front_query = pid_query[rand_idx]
                        new_query.append(tuple(front_query))
                        # random sample the back one from gallery
                        rand_idx = random.choice(back_gallery)
                        back_gallery = pid_gallery[rand_idx]
                        new_gallery.append(tuple(back_gallery))
                        front_in_query.append(pid)
                else:
                    if (len(front_gallery) > 0 and len(back_query) > 0):
                        # random sample the back one from query
                        rand_idx = random.choice(back_query)
                        back_query = pid_query[rand_idx]
                        new_query.append(tuple(back_query))
                        # random sample the front one from gallery
                        rand_idx = random.choice(front_gallery)
                        front_gallery = pid_gallery[rand_idx]
                        new_gallery.append(tuple(front_gallery))
                        back_in_query.append(pid)
                    else:
                        # abandon the pid
                        passed_pid.append(pid)
        num_avaliable_pid = len(all_pid) - len(passed_pid)
        front_to_be_make = num_avaliable_pid // 2 - len(front_in_query)
        back_to_be_make = num_avaliable_pid // 2 - len(back_in_query)
        rand_idx_list = [1] * front_to_be_make + [-1] * back_to_be_make
        if len(rand_idx_list) < len(free_pid):
            rand_idx_list += [-1] * (len(free_pid) - len(rand_idx_list))
        random.shuffle(rand_idx_list)

        for i, pid in enumerate(free_pid):
            pid_query = query[query_pid_array == pid]
            pid_gallery = gallery[gallery_pid_array == pid]
            pose_dirs_q = [pose_dict[x[0].split(dataset_name)[1][1:]] for x in pid_query]
            pose_dirs_q = np.array(pose_dirs_q)
            pose_dirs_g = [pose_dict[x[0].split(dataset_name)[1][1:]] for x in pid_gallery]
            pose_dirs_g = np.array(pose_dirs_g)
            from_query = np.where(pose_dirs_q == rand_idx_list[i])[0]
            rand_idx = random.choice(from_query)
            from_query = pid_query[rand_idx].copy()
            new_query.append(tuple(from_query))
            from_gallery = np.where(pose_dirs_g == -rand_idx_list[i])[0]
            rand_idx = random.choice(from_gallery)
            from_gallery = pid_gallery[rand_idx].copy()
            new_gallery.append(tuple(from_gallery))

            if rand_idx_list[i] == 1:
                front_in_query.append(pid)
            else:
                back_in_query.append(pid)
                
        print("passed pids: ", passed_pid, "num of front_in_query: ", len(front_in_query), "num of back_in_query: ", len(back_in_query))
        
        return new_query, new_gallery
    
    def modify_train_with_pose(self, train):
        dataset_name = train[0][1].split('_')[0]
        dataset_path = os.path.join(train[0][0].split(dataset_name)[0], dataset_name)
        pose_path = os.path.join(dataset_path, 'posture.txt')
        pose_dict = {}
        with open(pose_path, 'r') as f:
            for line in f:
                pose_dict[line.split(' ')[0]] = int(line.split(' ')[1])
        
        # sort train by pid
        train = sorted(train, key=lambda x: int(x[1].split('_')[1]))
        new_train = []
        init_pid = int(train[0][1].split('_')[1])
        end_pid = int(train[-1][1].split('_')[1])
        new_pid = cur_pid = init_pid
        i = 0
        while i < len(train):
            back_set = []
            front_set = []
            while int(train[i][1].split('_')[1]) == cur_pid:
                # pose_dict[cur_data] == -1, 0 -> back_set
                # pose_dict[cur_data] == 1, 0 -> front_set
                pose_dir = pose_dict[train[i][0].split(dataset_name)[1][1:]]
                if pose_dir == 0:
                    back_set.append(train[i])
                    front_set.append(train[i])
                elif pose_dir == -1:
                    back_set.append(train[i])
                elif pose_dir == 1:
                    front_set.append(train[i])
                else:
                    raise ValueError("pose_dir should be -1, 0, 1")
                i += 1
                if i == len(train):
                    break
            if len(back_set) > 0:
                back_set = [(x[0], dataset_name + '_' + str(new_pid), x[2]) for x in back_set]
                new_train.extend(back_set)
                new_pid += 1
            if len(front_set) > 0:
                front_set = [(x[0], dataset_name + '_' + str(new_pid), x[2]) for x in front_set]
                new_train.extend(front_set)
                new_pid += 1
            # if len(front_set) + len(back_set) == 0:
            #     print("no data for pid: ", cur_pid)
            cur_pid += 1
            
        return new_train
                
        
        

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        if len(data[0]) > 3:
            for _, pid, camid, _ in data:
                pids.add(pid)
                cams.add(camid)
        else:
            for _, pid, camid in data:
                pids.add(pid)
                cams.add(camid)
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        def _combine_data(data):
            for img_path, pid, camid, _ in data:
                if pid in self._junk_pids:
                    continue
                pid = self.dataset_name + "_" + str(pid)
                combined.append((img_path, pid, camid))

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not os.path.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        msg = '  ----------------------------------------\n' \
              '  subset   | # ids | # items | # cameras\n' \
              '  ----------------------------------------\n' \
              '  train    | {:5d} | {:7d} | {:9d}\n' \
              '  query    | {:5d} | {:7d} | {:9d}\n' \
              '  gallery  | {:5d} | {:7d} | {:9d}\n' \
              '  ----------------------------------------\n' \
              '  items: images/tracklets for image/video dataset\n'.format(
            num_train_pids, len(self.train), num_train_cams,
            num_query_pids, len(self.query), num_query_cams,
            num_gallery_pids, len(self.gallery), num_gallery_cams
        )

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.
    All other image datasets should subclass it.
    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def show_train(self):
        logger = logging.getLogger('PAT')
        num_train_pids, num_train_cams = self.parse_data(self.train)
        logger.info('=> Loaded {}'.format(self.__class__.__name__))
        logger.info('  ----------------------------------------')
        logger.info('  subset   | # ids | # images | # cameras')
        logger.info('  ----------------------------------------')
        logger.info('  train    | {:5d} | {:8d} | {:9d}'.format(num_train_pids, len(self.train), num_train_cams))
        logger.info('  ----------------------------------------')

    def show_test(self):
        logger = logging.getLogger('PAT')
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)
        logger.info('=> Loaded {}'.format(self.__class__.__name__))
        logger.info('  ----------------------------------------')
        logger.info('  subset   | # ids | # images | # cameras')
        logger.info('  ----------------------------------------')
        logger.info('  query    | {:5d} | {:8d} | {:9d}'.format(num_query_pids, len(self.query), num_query_cams))
        logger.info('  gallery  | {:5d} | {:8d} | {:9d}'.format(num_gallery_pids, len(self.gallery), num_gallery_cams))
        logger.info('  ----------------------------------------')
