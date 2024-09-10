from __future__ import print_function

import copy
import warnings

import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """Plots the population's average and best fitness."""
    if plt is None:
        warnings.warn("Matplotlib is not available. Cannot plot statistics.")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="Average")
    plt.plot(generation, avg_fitness - stdev_fitness, 'g--', label="-1 SD")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g--', label="+1 SD")
    plt.plot(generation, best_fitness, 'r-', label="Best")

    plt.title("Population's Average and Best Fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.legend(loc="best")
    if ylog:
        plt.yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """Plots the trains for a single spiking neuron."""
    t_values = [t for t, I, v, u, f in spikes]
    v_values = [v for t, I, v, u, f in spikes]
    u_values = [u for t, I, v, u, f in spikes]
    I_values = [I for t, I, v, u, f in spikes]
    f_values = [f for t, I, v, u, f in spikes]

    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    axs[0].plot(t_values, v_values, 'g-')
    axs[0].set_ylabel("Potential (mV)")
    axs[0].grid(True)

    axs[1].plot(t_values, f_values, 'r-')
    axs[1].set_ylabel("Fired")
    axs[1].grid(True)

    axs[2].plot(t_values, u_values, 'r-')
    axs[2].set_ylabel("Recovery (u)")
    axs[2].grid(True)

    axs[3].plot(t_values, I_values, 'r-o')
    axs[3].set_ylabel("Current (I)")
    axs[3].set_xlabel("Time (ms)")
    axs[3].grid(True)

    if title:
        plt.suptitle(f"Izhikevich's Spiking Neuron Model ({title})")
    else:
        plt.suptitle("Izhikevich's Spiking Neuron Model")

    if filename:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()

    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """Visualizes speciation throughout evolution."""
    if plt is None:
        warnings.warn("Matplotlib is not available. Cannot plot species statistics.")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.stackplot(range(num_generations), *curves, labels=[f"Species {i+1}" for i in range(len(curves))])

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")
    plt.legend(loc="best")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg'):
    """Draws a neural network with arbitrary topology from a genome."""
    if graphviz is None:
        warnings.warn("Graphviz is not available. Cannot draw the network.")
        return

    node_names = node_names or {}
    node_colors = node_colors or {}

    node_attrs = {
        'shape': 'circle',
        'fontsize': '9',
        'height': '0.2',
        'width': '0.2'}

    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)

    inputs = set(config.genome_config.input_keys)
    outputs = set(config.genome_config.output_keys)

    for k in inputs:
        name = node_names.get(k, str(k))
        input_attrs = {'style': 'filled', 'shape': 'box', 'fillcolor': node_colors.get(k, 'lightgray')}
        dot.node(name, _attributes=input_attrs)

    for k in outputs:
        name = node_names.get(k, str(k))
        node_attrs = {'style': 'filled', 'fillcolor': node_colors.get(k, 'lightblue')}
        dot.node(name, _attributes=node_attrs)

    if prune_unused:
        connections = {(cg.in_node_id, cg.out_node_id) for cg in genome.connections.values() if cg.enabled or show_disabled}
        used_nodes = set(outputs)
        pending = set(outputs)
        while pending:
            new_pending = set()
            for a, b in connections:
                if b in pending and a not in used_nodes:
                    new_pending.add(a)
                    used_nodes.add(a)
            pending = new_pending
    else:
        used_nodes = set(genome.nodes.keys())

    for n in used_nodes:
        if n not in inputs and n not in outputs:
            attrs = {'style': 'filled', 'fillcolor': node_colors.get(n, 'white')}
            dot.node(str(n), _attributes=attrs)

    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_name = node_names.get(cg.in_node_id, str(cg.in_node_id))
            output_name = node_names.get(cg.out_node_id, str(cg.out_node_id))
            style = 'solid' if cg.enabled else 'dotted'
            color = 'green' if cg.weight > 0 else 'red'
            width = str(0.1 + abs(cg.weight / 5.0))
            dot.edge(input_name, output_name, _attributes={'style': style, 'color': color, 'penwidth': width})

    dot.render(filename, view=view)

    return dot
