# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:44:27 2016

@author: yungkuo
"""

import numpy as np
import os
import scipy.optimize as opt
from lmfit.models import PolynomialModel
import scipy.ndimage as ndi


def listdir(path, startswith, endswith):
    for f in os.listdir(path):
        if f.startswith(startswith):
            if f.endswith(endswith):
                yield f

def get_roi_square(point, pad):
    """Return a square selection of pixels around `point`.
    """
    col, row = point
    col = int(col)
    row = int(row)
    mask = (slice(row-pad[0], row+pad[0]+1), slice(col-pad[1], col+pad[1]+1))
    return mask

def get_roi_square_3d(point, pad):
    """Return a square selection of pixels around `point`.
    """
    col, row = point
    col = int(col)
    row = int(row)
    mask = (slice(None, None), slice(row-pad[0], row+pad[0]+1), slice(col-pad[1], col+pad[1]+1))
    return mask

def get_local_max(image, point, pad):
    roi = get_roi_square(point, pad)
    col, row = point.astype('int')
    while image[row, col] != image[roi].max():
        row = int(row-pad[0]+np.where(image[roi]==image[roi].max())[0])
        col = int(col-pad[1]+np.where(image[roi]==image[roi].max())[1])
        point = (col,row)
        roi = get_roi_square(point, pad)
    return point


def get_background(movie, pts, scan):
    bg_indx= np.ones(movie.shape[1])
    for i in range(len(pts[:,1])):
        bg_indx[int(pts[i,1])-scan[0] : int(pts[i,1])+scan[0]] = 0
    bg_mask = np.tile(bg_indx,(movie.shape[2],1)).T
    bg = movie*bg_mask
    return bg

def get_stack_image(movie, pt, pad):
    roi = get_roi_square_3d(pt, pad)
    boxmovie = movie[roi]
    stack_image = np.reshape(boxmovie, (movie.shape[0]*(2*pad[0]+1),2*pad[1]+1)).T
    return stack_image



def get_timetrace(movie, pt, pad):
    roi = get_roi_square_3d(pt, pad)
    tt = movie[roi].mean(1).mean(1)
    return tt

def get_Von_spec(movie, pt, pad, threshold):
    roi = get_roi_square_3d(pt, pad)
    tt = get_timetrace(movie, pt, pad)
    movie = movie[roi]
    movie = movie[::2]
    movie = np.array([movie[i] for i in range(len(movie)) if tt[::2][i] > threshold])
    spec = movie.mean(1).mean(0)
    return spec

def get_Voff_spec(movie, pt, pad, threshold):
    roi = get_roi_square_3d(pt, pad)
    tt = get_timetrace(movie, pt, pad)
    movie = movie[roi]
    movie = movie[1::2]
    movie = np.array([movie[i] for i in range(len(movie)) if tt[1::2][i] > threshold])
    spec = movie.mean(1).mean(0)
    return spec

def get_peak_ccms(movie, pt, pad, x):
    roi = get_roi_square_3d(pt, pad)
    tt = get_timetrace(movie, pt, pad)
    movie = movie[roi]
    spec_series = movie.mean(1)
    peaks = np.sum(spec_series*x, axis=1)/np.sum(spec_series,axis=1)
    return peaks
    
def finddot(image, scan, nstd):
    pts = []
    for i in range(scan[0], image.shape[0]-scan[0], 1):
        for j in range(scan[1], image.shape[1]-scan[1], 1):
            roi = get_roi_square([j,i],scan)
            if image[i,j] == np.max(image[roi]):
                roi_big = get_roi_square([j,i], [scan[0]*4,scan[1]*4])
                if np.mean(image[roi]) > np.mean(image[roi_big])+nstd*np.std(image[roi_big], ddof=1):
                    pt = [j,i]
                    pts = np.append(pts, pt)
    return np.reshape(pts,[len(pts)//2,2])


def findparticle(image, boundary, scan, nstd):
    pts = []
    for i in np.arange(scan[0],image.shape[0]-scan[0],1, dtype='int'):
        for j in np.arange(boundary[0],boundary[1],1, dtype='int'):
            if image[i,j] == np.max(image[(i-scan[0]):(i+scan[0]),(j-scan[1]):(j+scan[1])]):
                #if (np.sum(image[(i-1):(i+2),(j-1):(j+2)])-image[i,j])/8 > (np.sum(image[(i-2):(i+3),(j-2):(j+3)])-np.sum(image[(i-1):(i+2),(j-1):(j+2)]))/16:
                if np.mean(image[(i-scan[0]):(i+scan[0]),(j-scan[1]):(j+scan[1])]) > np.mean(image[(i-scan[0]):(i+scan[0]),boundary[0]:boundary[1]])+nstd*np.std(image[(i-scan[0]):(i+scan[0]),boundary[0]:boundary[1]], ddof=1):
                    pt = [j,i]
                    pts = np.append(pts, pt)
    return np.reshape(pts,[len(pts)/2,2])


def mask3d(refimg, pts, scan):
    local = refimg[:,pts[1]-scan[0]:pts[1]+scan[0],pts[0]-scan[1]:pts[0]+scan[1]]
    return local

def shift_match(point_prism, point_dot):
    def func(p):
        return np.sum((point_dot[:,1]-(point_prism[:,1]*p[0]-p[1]))**2)
    res = opt.minimize(func, x0=np.array([1.,0.]), method='SLSQP')
    if res.success:
        magn = res.x[0]
        shift = res.x[1]
    return magn, shift

def get_blinking_threshold(movie, pt, pad, iterations_to_find_threshold, nstd):
    tt = get_timetrace(movie, pt, pad)
    threshold = np.mean(tt)
    if iterations_to_find_threshold != 0:
        off_mean = np.mean([tt[i] for i in range(len(tt)) if tt[i] < threshold])
        off_std = np.std([tt[i] for i in range(len(tt)) if tt[i] < threshold], ddof=1,dtype='d')
        threshold1 = off_mean + off_std*nstd
        for i in range(iterations_to_find_threshold):
            if threshold1 != threshold:
                threshold = threshold1
                off_mean = np.mean([tt[i] for i in range(len(tt)) if tt[i] < threshold])
                off_std = np.std([tt[i] for i in range(len(tt)) if tt[i] < threshold], ddof=1,dtype='d')
                threshold1 = off_mean + off_std*nstd
    return threshold


def get_Gauss_filtered_image(image, sigma):
    data = image.shape
    image_LPed = np.zeros(data)
    for i in range(data[0]):
        image_LPed[i,:] = ndi.gaussian_filter(image[i,:], sigma)
    image_HPed = image-image_LPed
    return image_LPed, image_HPed


