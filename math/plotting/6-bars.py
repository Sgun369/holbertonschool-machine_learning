#!/usr/bin/env python3
"""Stacking Bars"""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # Assign colors to each fruit based on the specification
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']
    fruit_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']
    person_labels = ['Farrah', 'Fred', 'Felicia']

    # Set up the bar positions (x-coordinates for each group of bars)
    x = np.arange(len(person_labels))

    # Plot bars
    # Start bottom at zero for the first row of fruit
    bottom = np.zeros(len(person_labels))
    for idx, (fruit_row, color) in enumerate(zip(fruit, colors)):
        plt.bar(
            x,
            fruit_row,
            color=color,
            label=fruit_labels[idx],
            bottom=bottom,
            width=0.5)
        bottom += fruit_row  # Update bottom position for the next stack

    # Add labels, title, and legend
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(x, person_labels)
    plt.yticks(np.arange(0, 81, 10))
    plt.legend(title="Fruit Type")

    # Show the plot
    plt.show()
