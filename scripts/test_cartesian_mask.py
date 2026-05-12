from sampling.cartesian import random_cartesian_mask, get_sampling_percentage


def main():
    mask = random_cartesian_mask(
        shape=(256, 256),
        acceleration=4,
        center_fraction=0.08,
        seed=42,
    )

    print("Mask shape:", mask.shape)
    print("Mask dtype:", mask.dtype)
    print("Unique values:", sorted(set(mask.flatten())))
    print("Sampling percentage:", get_sampling_percentage(mask))


if __name__ == "__main__":
    main()
