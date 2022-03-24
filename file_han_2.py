#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 08:38:32 2021

@author: raj
"""

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
                        shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'train'))

                    elif file == f'{i}-bg-0{j}-{k}.png':
                        shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'bg'))

                    elif file == f'{i}-cl-0{j}-{k}.png':
                        shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'cl'))
                for l in range(5,7):
                    for m in angle:
                        print(f'{file}, {i}, {l}, {m}')
                        print(f'{i}-nm-0{l}-{m}.png')
                        if file == f'{i}-nm-0{l}-{m}.png':
                            shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'test'))
                        
                        elif file == f'{i}-bg-0{l}-{m}.png':
                            shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'bg'))
                            
                        elif file == f'{i}-cl-0{l}-{m}.png':
                            shutil.copy(os.path.join(cur_dir,file) , os.path.join(new_dir ,'cl'))

                            


 