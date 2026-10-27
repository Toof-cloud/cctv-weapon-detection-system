# Dataset Cleaning Log

## Dataset Cleaning Overview

Following the manual review of 500 sampled images from the Open Images V6 dataset, non-ideal samples were identified and isolated from the training dataset. Images determined to be toy weapons, artwork, posters, statues, display models, invalid annotations, and other non-representative samples were moved to a quarantine directory rather than permanently deleted.

Quarantined images were stored in:

removed_images/
├── handgun/
└── knife/

This approach preserves the original dataset and allows future re-evaluation of removed samples if necessary.

---

# Handgun Dataset

## Reviewer A

Removed Categories:

- Toy Guns
- Artwork / Paintings
- Statues

## Reviewer B

Removed Categories:

- Toy Guns
- LEGO Guns
- Prop Guns
- Posters
- Artwork
- Display Models
- Statues

## Reviewer C

Removed Categories:

- Toy Guns
- Artwork / Paintings
- Statues / Display Models

## Reviewer D

Removed Categories:

- Toy Guns
- LEGO Guns
- Prop Guns
- Posters
- 3D Renderings
- Display Models
- Statue-Based Artwork

### Handgun Cleaning Status

- Invalid handgun IDs identified and documented
- Invalid handgun samples quarantined
- Current quarantined handgun images: 32

Status: Complete

---

# Knife Dataset

## Reviewer A

Removed Categories:

- Picture / Artwork of Knife

## Reviewer B

Removed Categories:

- None Identified

## Reviewer C

Removed Categories:

- Poster / Screen
- Drawing / Cartoon
- Invalid Annotation

## Reviewer D

Removed Categories:

- None Identified

### Knife Cleaning Status

- Invalid knife IDs identified and documented
- Current quarantined knife images: 26

Pending Retrieval:

- 1388df78ddce18ce
- 1909fcdcd54d91e

These images were identified during review but were unavailable in the local dataset copy at the time of cleaning.

Status: Complete (Pending Retrieval of 2 Missing Images)

---

# Cleaning Methodology

Images were quarantined when they contained:

## Handguns

- Toy Guns
- LEGO Guns
- Prop Guns
- Posters
- Paintings
- Artwork
- Statues
- Display Models
- 3D Renderings
- Wrong Weapon Types

## Knives

- Artwork
- Posters
- Screen Captures
- Drawings
- Cartoons
- Invalid Annotations
- Images with no clearly visible knife object

Partially visible but legitimate weapons were retained because partial visibility is common in real-world CCTV footage.

---

# Cleaning Status Summary

Dataset Review: ✅ Complete

Handgun Dataset Cleaning: ✅ Complete

Knife Dataset Cleaning: ✅ Complete (Pending Retrieval of 2 Missing Images)

Dataset Preparation Phase: ✅ Complete