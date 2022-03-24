#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 20:27:51 2021

@author: raj
"""

import os
import shutil
cur_dir = '/home/raj/GAIT/GEI'
new_dir = '/home/raj/new_gait'
angle = ['0','18','36','54','72','90','108','126','144','162','180']
for files in os.walk(cur_dir): 
    for file in files[2]:
        for i in range(1,125):
            for j in range(1,5):
                for k in angle:
                    print(f'{file}, {i}, {j}, {k}')
                    print(f'{i}-nm-0{j}-{k}.png')
                    if file == f'{i}-nm-0{j}-{k}.png':
                        print('-----------')
                        shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'train'))
                        print('Copied to train')
                for l in range(5,7):
                    for m in angle:
                        print(f'{file}, {i}, {l}, {m}')
                        print(f'{i}-nm-0{l}-{m}.png')
                        if file == f'{i}-nm-0{l}-{m}.png':
                            print('**********')
                            shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'test'))
                            print('Copied to test')
                for n in range(1,7):
                    for o in angle:
                        print(f'{file}, {i}, {n}, {o}')
                        if file == f'{i}-bg-0{n}-{o}.png':
                            print('////////////')
                            shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'bg'))
                            print('Copied to bg')
                            
                        if file == f'{i}-cl-0{n}-{o}.png':
                            print('$$$$$$$$$$$$')
                            shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'cl'))
                            print('Copied to cl')                    
                        