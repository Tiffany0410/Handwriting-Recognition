/*
 * This code is provided solely for the personal and private use of students
 * taking the CSC209H course at the University of Toronto. Copying for purposes
 * other than this use is expressly prohibited. All forms of distribution of
 * this code, including but not limited to public repositories on GitHub,
 * GitLab, Bitbucket, or any other online platform, whether as given or with
 * any changes, are expressly prohibited.
 *
 * Authors: Mustafa Quraish, Bianca Schroeder, Karen Reid
 *
 * All of the files in this directory and all subdirectories are:
 * Copyright (c) 2021 Karen Reid
 */

#include "dectree.h"

/**
 * Load the binary file, filename into a Dataset and return a pointer to 
 * the Dataset. The binary file format is as follows:
 *
 *     -   4 bytes : `N`: Number of images / labels in the file
 *     -   1 byte  : Image 1 label
 *     - NUM_PIXELS bytes : Image 1 data (WIDTHxWIDTH)
 *          ...
 *     -   1 byte  : Image N label
 *     - NUM_PIXELS bytes : Image N data (WIDTHxWIDTH)
 *
 * You can set the `sx` and `sy` values for all the images to WIDTH. 
 * Use the NUM_PIXELS and WIDTH constants defined in dectree.h
 */
Dataset *load_dataset(const char *filename) {
    Dataset *ds = malloc(sizeof(Dataset)); 
    FILE *data_file;
    int num_images;

    data_file = fopen(filename, "rb");
    if (data_file != NULL){
        // read in num_images
        fread(&num_images, sizeof(int), 1, data_file);

        unsigned char *labels = malloc(sizeof(unsigned char)*num_images);
        Image *images = malloc(sizeof(Image)*num_images);

        for (int i = 0; i < num_images; i++){
            // store labels
            unsigned char label;
            fread(&label, sizeof(unsigned char), 1, data_file);
            labels[i] = label;
            
            // construct Image
            Image image;
            image.sx = WIDTH;
            image.sy = WIDTH;
            unsigned char *data = malloc(sizeof(unsigned char)*(NUM_PIXELS));
            fread(data, sizeof(unsigned char), NUM_PIXELS, data_file);

            image.data = data;
            images[i] = image;
        }

        ds->num_items = num_images;
        ds->images = images;
        ds->labels = labels;
        fclose(data_file);
    }
    else{
        perror("fopen");
        exit(1);
    }

    return ds;
}

/**
 * Compute and return the Gini impurity of M images at a given pixel
 * The M images to analyze are identified by the indices array. The M
 * elements of the indices array are indices into data.
 * This is the objective function that you will use to identify the best 
 * pixel on which to split the dataset when building the decision tree.
 *
 * Note that the gini_impurity implemented here can evaluate to NAN 
 * (Not A Number) and will return that value. Your implementation of the 
 * decision trees should ensure that a pixel whose gini_impurity evaluates 
 * to NAN is not used to split the data.  (see find_best_split)
 * 
 * DO NOT CHANGE THIS FUNCTION; It is already implemented for you.
 */
double gini_impurity(Dataset *data, int M, int *indices, int pixel) {
    int a_freq[10] = {0}, a_count = 0;
    int b_freq[10] = {0}, b_count = 0;

    for (int i = 0; i < M; i++) {
        int img_idx = indices[i];

        // The pixels are always either 0 or 255, but using < 128 for generality.
        if (data->images[img_idx].data[pixel] < 128) {
            a_freq[data->labels[img_idx]]++;
            a_count++;
        } else {
            b_freq[data->labels[img_idx]]++;
            b_count++;
        }
    }

    double a_gini = 0, b_gini = 0;
    for (int i = 0; i < 10; i++) {
        double a_i = ((double)a_freq[i]) / ((double)a_count);
        double b_i = ((double)b_freq[i]) / ((double)b_count);
        a_gini += a_i * (1 - a_i);
        b_gini += b_i * (1 - b_i);
    }

    // Weighted average of gini impurity of children
    return (a_gini * a_count + b_gini * b_count) / M;
}

/**
 * Given a subset of M images and the array of their corresponding indices, 
 * find and use the last two parameters (label and freq) to store the most
 * frequent label in the set and its frequency.
 *
 * - The most frequent label (between 0 and 9) will be stored in `*label`
 * - The frequency of this label within the subset will be stored in `*freq`
 * 
 * If multiple labels have the same maximal frequency, return the smallest one.
 */
void get_most_frequent(Dataset *data, int M, int *indices, int *label, int *freq) {
    // TODO: Set the correct values and return
    int freqs[10] = {0};

    for (int i = 0; i < M; i++){
        int img_idx = indices[i];
        int l = data->labels[img_idx];
        freqs[l]++;
    }

    int best_label = 0;
    // Find highest frequency
    for (int i = 0; i < 10; i++){
        if (freqs[i] > freqs[best_label]){
            best_label = i;
        }
    }
    *label = best_label;
    *freq = freqs[best_label];
    return;
}

/**
 * Given a subset of M images as defined by their indices, find and return
 * the best pixel to split the data. The best pixel is the one which
 * has the minimum Gini impurity as computed by `gini_impurity()` and 
 * is not NAN. (See handout for more information)
 * 
 * The return value will be a number between 0-783 (inclusive), representing
 *  the pixel the M images should be split based on.
 * 
 * If multiple pixels have the same minimal Gini impurity, return the smallest.
 */
int find_best_split(Dataset *data, int M, int *indices) {
    // TODO: Return the correct pixel
    int smallest = 0;
    double ginis[784] = {0};

    for (int i = 0; i < 784; i++){
        double gini = gini_impurity(data, M, indices, i);
        ginis[i] = gini;
    }

    // Find smallest gini
    int j = 0;
    while (isnan(ginis[j]) == 1){
        j++;
    }
    
    smallest = j;

    for (int i = j; i < 784; i++){
        if (ginis[i] < ginis[smallest]){
            smallest = i;
        }
    }
    return smallest;
}

/**
 * Create the Decision tree. In each recursive call, we consider the subset of the
 * dataset that correspond to the new node. To represent the subset, we pass 
 * an array of indices of these images in the subset of the dataset, along with 
 * its length M. Be careful to allocate this indices array for any recursive 
 * calls made, and free it when you no longer need the array. In this function,
 * you need to:
 *
 *    - Compute ratio of most frequent image in indices, do not split if the
 *      ration is greater than THRESHOLD_RATIO
 *    - Find the best pixel to split on using `find_best_split`
 *    - Split the data based on whether pixel is less than 128, allocate 
 *      arrays of indices of training images and populate them with the 
 *      subset of indices from M that correspond to which side of the split
 *      they are on
 *    - Allocate a new node, set the correct values and return
 *       - If it is a leaf node set `classification`, and both children = NULL.
 *       - Otherwise, set `pixel` and `left`/`right` nodes 
 *         (using build_subtree recursively). 
 */
DTNode *build_subtree(Dataset *data, int M, int *indices) {
    // TODO: Construct and return the tree
    DTNode *dtnode = malloc(sizeof(DTNode));

    // Compute ratio of most frequent image in indices
    int *freq = malloc(sizeof(int));
    int *label = malloc(sizeof(int));

    get_most_frequent(data, M, indices, label, freq);

    // Do not split if the ratio is greater than THRESHOLD_RATIO
    if (M > 0){
        if ((float) *freq / M >= THRESHOLD_RATIO){
            dtnode->pixel = -1;
            dtnode->classification = *label;
            dtnode->left = NULL;
            dtnode->right = NULL;
        }
        else{
            // Find the best pixel to split on using `find_best_split`
            int pixel = find_best_split(data, M, indices);

            int right_M = 0;
            int left_M = 0;
            for (int i = 0; i < M; i ++){
                if (data->images[indices[i]] .data[pixel] >= 128){
                    right_M++;
                }
                else{
                    left_M++;
                }
            }

            int *right_indices = malloc(sizeof(int) * right_M);
            int *left_indices = malloc(sizeof(int) * left_M);

            int right_curr = 0;
            int left_curr = 0;

            for (int i = 0; i < M; i ++){
                if (data->images[indices[i]].data[pixel] >= 128){
                    right_indices[right_curr] = indices[i];
                    right_curr++;
                }
                else{
                    left_indices[left_curr] = indices[i];
                    left_curr++;
                }
            }

            // Check empty
            if (right_M == 0){
                dtnode->right = NULL;
            }
            else{
                dtnode->right = build_subtree(data, right_M, right_indices);
            }

            if (left_M == 0){
                dtnode->left = NULL;
            }
            else{
                dtnode->left = build_subtree(data, left_M, left_indices);
            }
            dtnode->classification = -1;
            dtnode->pixel = pixel;
            free(right_indices);
            free(left_indices);
        }
    }

    free(freq);
    free(label);
    return dtnode;
}

/**
 * This is the function exposed to the user. All you should do here is set
 * up the `indices` array correctly for the entire dataset and call 
 * `build_subtree()` with the correct parameters.
 */
DTNode *build_dec_tree(Dataset *data) {
    // TODO: Set up `indices` array, call `build_subtree` and return the tree.
    // HINT: Make sure you free any data that is not needed anymore
    int size = data->num_items;
    int array[size];
    for (int i = 0; i < size; i++){
        array[i] = i;
    }
    DTNode *dtnode = build_subtree(data, size, array);
    return dtnode;
}

/**
 * Given a decision tree and an image to classify, return the predicted label.
 */
int dec_tree_classify(DTNode *root, Image *img) {
    // TODO: Return the correct label
    int classify = root->classification;
    DTNode *dtnode = root;
    while (classify == -1){
        int c = dtnode->classification;
        if (c != -1){
            return c;
        }
        int pixel = dtnode->pixel;
        int img_pixel = img->data[pixel];
        if (img_pixel >= 128){
            dtnode = dtnode->right;
        }
        else{
            dtnode = dtnode->left;
        }
    }

    return classify;
}

/**
 * This function frees the Decision tree.
 */
void free_dec_tree(DTNode *node) {
    if (node->right != NULL){
        free_dec_tree(node->right);
        node->right = NULL;
    }
    if (node->left != NULL){
        free_dec_tree(node->left);
        node->left = NULL;
    }
    if (node->left == NULL && node->right == NULL){
        free(node);
    }
    return;
}

/**
 * Free all the allocated memory for the dataset
 */
void free_dataset(Dataset *data) {
    // TODO: Free dataset (Same as A1)
    for (int i = 0; i < data->num_items; i++){
        free(data->images[i].data);
    }
    free(data->images);
    free(data->labels);
    free(data);
    return;
}

