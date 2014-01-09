/*
 * $Id$
 * $URL$
 * ---------------------------------------------------------------------
 * This file is part of Simulation Core Library, a Java-based library
 * for efficient numerical simulation of biological models.
 *
 * Copyright (C) 2007-2014 jointly by the following organizations:
 * 1. University of Tuebingen, Germany
 * 2. Keio University, Japan
 * 3. Harvard University, USA
 * 4. The University of Edinburgh, UK
 * 5. EMBL European Bioinformatics Institute (EBML-EBI), Hinxton, UK
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation. A copy of the license
 * agreement is provided in the file named "LICENSE.txt" included with
 * this software distribution and also available online as
 * <http://www.gnu.org/licenses/lgpl-3.0-standalone.html>.
 * ---------------------------------------------------------------------
 */
package org.simulator.sbml.astnode;

import org.sbml.jsbml.ASTNode;
import org.sbml.jsbml.SpeciesReference;
import org.simulator.sbml.SBMLValueHolder;

/**
 * This class computes and stores values of {@link ASTNode}s that refer to a
 * {@link SpeciesReference}.
 * 
 * @author Roland Keller
 * @version $Rev$
 * @since 1.0
 */
public class SpeciesReferenceValue extends ASTNodeValue {

	/**
	 * The value holder that stores the current simulation values
	 */
	private SBMLValueHolder valueHolder;

	/**
	 * The id of the species reference
	 */
	private String id;

	/**
	 * 
	 * @param interpreter
	 * @param node
	 * @param sr
	 * @param valueHolder
	 */
	public SpeciesReferenceValue(ASTNodeInterpreter interpreter,
			ASTNode node, SpeciesReference sr, SBMLValueHolder valueHolder) {
		super(interpreter, node);
		this.id=sr.getId();
		this.valueHolder=valueHolder;
	}

	/* (non-Javadoc)
	 * @see org.simulator.sbml.astnode.ASTNodeValue#computeDoubleValue()
	 */
	@Override
	protected void computeDoubleValue(double delay) {
		doubleValue = valueHolder.getCurrentStoichiometry(id);
	}

}
