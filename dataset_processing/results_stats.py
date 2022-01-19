import os
import matplotlib.pyplot as plt
import numpy as np


def create_files_registry(filename, start_folder="."):
    if not os.path.exists(filename):
        with open(filename, "w+") as folders_file:
            for root, dirs, files in os.walk(start_folder):
                for name in files:
                    if name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith(".png"):
                        folders_file.write(os.path.join(root, name) + '\n')

                        
def extract_ages_for_race(files):
    ages = []
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png")):
            ages.append(int(file.split('_')[0]))
    ages.sort()
    return ages


def split_dataset_into_races(root):
    races = [[], [], [], [], []]    # White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern)
    
    for _, _, files in os.walk(root):
        for filename in files:
            if filename.endswith((".jpg", ".jpeg", ".png")):
                descriptions = filename.split('_')
                if len(descriptions) > 3:  # in dataset are errors in filenames
                    race = int(descriptions[2])
                    if not race > 4:
                        races[race].append(filename)
    
    return races


def plot_age_stats(hist_data):
    x_values = list(range(0,120,10))

    nrows = 3
    ncols = 2
    
    plt.style.use('Solarize_Light2')
    fig, axes = plt.subplots(nrows = nrows, ncols = ncols, figsize = (9, 6), sharey = True)
    fig.suptitle("UTKFace races histogram", fontweight ='bold', fontsize = 18)
    
    
    races = ['White', 'Black', 'Asian', 'Indian', 'Other']
    index = 0
    
    for i in range(nrows):
        for j in range(ncols):
            if i == 2 and j == 1:
                axes[i][j].axis('off')
            else:
                axes[i][j].set_xticks(x_values)
                axes[i][j].hist(hist_data[index], 100, histtype='bar')
                axes[i][j].set_title(races[index] + ' people ages')
            index += 1

    fig.tight_layout()

    plt.show()


def display_people_histograms():
    races = split_dataset_into_races("./results/datasety/humans")
    
    hist_data = []
    
    for race in races:
        hist_data.append(extract_ages_for_race(race))
    
    plot_age_stats(hist_data)
    

def count_images_in_folder(folder):
    directories = next(os.walk(folder))[1]
    if len(directories) == 0:
        return {folder.split('/')[-1]: len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])}
    else:
        results = {}
        directories_full_paths = [os.path.join(folder, directory) for directory in directories]
        for i in range(len(directories)):
            results[directories[i]] = len([name for name in os.listdir(directories_full_paths[i]) if os.path.isfile(os.path.join(directories_full_paths[i], name))])
        return results
        
    
    
def display_animals_column_plots():
    animals = count_images_in_folder('./datasety/animals_10')
    recognised_animals = count_images_in_folder('./results/datasety/animals_10')
    chimpanzees = count_images_in_folder('./datasety/chimpanzees')
    recognised_chimpanzees = count_images_in_folder('./results/datasety/chimpanzees')
    monkeys = count_images_in_folder('./datasety/monkeys')
    recognised_monkeys = count_images_in_folder('./results/datasety/monkeys')
    
    # set width of bar
    barWidth = 0.25
    plt.style.use('Solarize_Light2')
    fig, axes = plt.subplots(nrows = 2, figsize = (9, 9))
    fig.suptitle("Comparison of initial datasets count vs recognised faces", fontweight ='bold', fontsize = 18)
    
    br1 = np.arange(len(animals) + len(chimpanzees))
    br2 = [x + barWidth for x in br1]
    
    axes[0].set_title('Animals_10 + Chimpanzees')
    axes[0].bar(br1, list(animals.values()) + list(chimpanzees.values()), color = 'b', 
                width = barWidth, edgecolor ='grey', 
                label = 'Monkeys + Chimpanzees - init')
    axes[0].bar(br2, list(recognised_animals.values()) + list(recognised_chimpanzees.values()), 
                color = 'y', width = barWidth, edgecolor ='grey', 
                label = 'Monkeys + Chimpanzees - recognised')
    axes[0].set_xlabel('Folder names')
    axes[0].set_ylabel('Image count')
    axes[0].set_xticks([r + (0.5 * barWidth) for r in range(len(animals) + len(chimpanzees))], 
                       list(animals.keys()) + list(chimpanzees.keys()))
    axes[0].legend()
    
    
    
    br3 = np.arange(len(monkeys))
    br4 = [x + barWidth for x in br3]

    axes[1].set_title('Monkeys')
    axes[1].bar(br3, monkeys.values(), color = 'r', width = barWidth, edgecolor ='grey', 
                label = 'Monkeys - init')
    axes[1].bar(br4, recognised_monkeys.values(), color = 'g', width = barWidth, edgecolor ='grey', 
                label = 'Monkeys - recognised')
    axes[1].set_xlabel('Folder names')
    axes[1].set_ylabel('Image count')
    axes[1].set_xticks([r + (0.5 * barWidth) for r in range(len(monkeys))], monkeys.keys())
    axes[1].legend()

    fig.tight_layout()
    plt.show()
    
    
    
    
    

def main():
    display_people_histograms()
    display_animals_column_plots()
    
    

if __name__ == "__main__":
   main()