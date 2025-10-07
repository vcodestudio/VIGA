# rag.py
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import requests
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None


class BlenderInfinigenRAG:
    """RAG tool: Search for bpy and Infinigen related documentation and generate code examples"""
    
    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            if not OPENAI_AVAILABLE:
                logging.warning("OpenAI library not available. Enhanced generation disabled.")
            elif not self.openai_api_key:
                logging.warning("OpenAI API key not provided. Enhanced generation disabled.")
        
        # Predefined documentation knowledge base
        self.knowledge_base = self._build_knowledge_base()
        
        # Instruction pattern matching
        self.instruction_patterns = {
            'physics_placement': [
                r'physics.*placement|rigid.*body|physics.*simulation',
                r'gravity|collision|rigidbody|physics.*world'
            ],
            'object_creation': [
                r'create.*object|add.*object|generate.*object',
                r'primitive|basic.*shape|mesh.*add|bpy\.ops\.mesh'
            ],
            'lighting': [
                r'lighting|illumination|light|shadow',
                r'material|texture|shader|node'
            ],
            'animation': [
                r'animation|keyframe|motion|transform',
                r'timeline|fcurve|driver|constraint'
            ],
            'scene_setup': [
                r'scene.*setup|environment|camera|render',
                r'world|background|composition'
            ]
        }
    
    def _build_knowledge_base(self) -> Dict:
        """Build knowledge base for bpy and Infinigen"""
        return {
            'bpy_physics': {
                'title': 'Blender Python Physics API',
                'apis': [
                    {
                        'name': 'bpy.ops.rigidbody.object_add',
                        'description': 'Add rigid body physics properties to object',
                        'example': 'bpy.ops.rigidbody.object_add(type=\'ACTIVE\')',
                        'use_case': 'Make object subject to physics laws'
                    },
                    {
                        'name': 'bpy.ops.rigidbody.world_add',
                        'description': 'Add physics world',
                        'example': 'bpy.ops.rigidbody.world_add()',
                        'use_case': 'Create physics simulation environment'
                    },
                    {
                        'name': 'bpy.context.scene.rigidbody_world.gravity',
                        'description': 'Set gravity',
                        'example': 'bpy.context.scene.rigidbody_world.gravity = (0, 0, -9.81)',
                        'use_case': 'Adjust gravity parameters'
                    }
                ]
            },
            'bpy_objects': {
                'title': 'Blender Python Object Creation API',
                'apis': [
                    {
                        'name': 'bpy.ops.mesh.primitive_cube_add',
                        'description': 'Create cube',
                        'example': 'bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))',
                        'use_case': 'Create basic geometry'
                    },
                    {
                        'name': 'bpy.ops.mesh.primitive_uv_sphere_add',
                        'description': 'Create sphere',
                        'example': 'bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location=(0, 0, 0))',
                        'use_case': 'Create spherical objects'
                    },
                    {
                        'name': 'bpy.ops.mesh.primitive_plane_add',
                        'description': 'Create plane',
                        'example': 'bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))',
                        'use_case': 'Create ground or platform'
                    }
                ]
            },
            'infinigen_physics': {
                'title': 'Infinigen Physics Utilities',
                'apis': [
                    {
                        'name': 'infinigen.core.placement.placement',
                        'description': 'Intelligent object placement system',
                        'example': 'placement.place_object(obj, surface, physics=True)',
                        'use_case': 'Physics-aware object placement'
                    },
                    {
                        'name': 'infinigen.core.placement.surface',
                        'description': 'Surface detection and placement',
                        'example': 'surface.find_surface_point(location, radius)',
                        'use_case': 'Place objects on surfaces'
                    },
                    {
                        'name': 'infinigen.core.physics.rigidbody',
                        'description': 'Rigid body physics setup',
                        'example': 'rigidbody.setup_rigidbody(obj, mass=1.0)',
                        'use_case': 'Set object physics properties'
                    }
                ]
            },
            'infinigen_scene': {
                'title': 'Infinigen Scene Generation',
                'apis': [
                    {
                        'name': 'infinigen.core.scene.scene',
                        'description': 'Scene generation and management',
                        'example': 'scene.add_objects(objects, placement_strategy="physics")',
                        'use_case': 'Create physics-realistic scenes'
                    },
                    {
                        'name': 'infinigen.core.lighting.lighting',
                        'description': 'Intelligent lighting setup',
                        'example': 'lighting.setup_natural_lighting(scene)',
                        'use_case': 'Set realistic lighting effects'
                    }
                ]
            }
        }
    
    def search_blender_docs(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Blender Python API documentation for relevant information"""
        try:
            # Search Blender API documentation
            blender_search_url = f"https://docs.blender.org/api/current/search.html?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(blender_search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            
            # Look for search results in the documentation
            search_results = soup.find_all('div', class_='search-results') or soup.find_all('div', class_='result')
            
            for result in search_results[:max_results]:
                title_elem = result.find('a') or result.find('h3')
                snippet_elem = result.find('p') or result.find('div', class_='highlight')
                
                if title_elem:
                    title = title_elem.get_text().strip()
                    url = title_elem.get('href', '')
                    if url and not url.startswith('http'):
                        url = f"https://docs.blender.org/api/current/{url}"
                    
                    snippet = ""
                    if snippet_elem:
                        snippet = snippet_elem.get_text().strip()
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'source': 'blender_docs'
                    })
            
            # If no specific search results, try to find relevant sections
            if not results:
                # Look for common API sections that might be relevant
                relevant_sections = self._find_relevant_blender_sections(query)
                results.extend(relevant_sections[:max_results])
            
            return results
            
        except Exception as e:
            logging.error(f"Blender docs search failed: {e}")
            return []

    def search_infinigen_docs(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Infinigen documentation for relevant information"""
        try:
            # Search Infinigen GitHub docs
            infinigen_search_url = f"https://github.com/princeton-vl/infinigen/tree/main/docs"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            response = requests.get(infinigen_search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            results = []
            
            # Look for documentation files and directories
            doc_links = soup.find_all('a', href=True)
            
            for link in doc_links:
                href = link.get('href', '')
                if '/docs/' in href and ('.md' in href or href.endswith('/')):
                    title = link.get_text().strip()
                    if title and title not in ['..', '.']:
                        # Extract relevant content based on query
                        if self._is_relevant_to_query(title, query):
                            results.append({
                                'title': f"Infinigen: {title}",
                                'url': f"https://github.com{href}",
                                'snippet': f"Documentation section: {title}",
                                'source': 'infinigen_docs'
                            })
            
            return results[:max_results]
            
        except Exception as e:
            logging.error(f"Infinigen docs search failed: {e}")
            return []

    def _find_relevant_blender_sections(self, query: str) -> List[Dict]:
        """Find relevant Blender API sections based on query keywords"""
        query_lower = query.lower()
        relevant_sections = []
        
        # Map query keywords to relevant API sections
        section_mapping = {
            'physics': [
                {'title': 'Rigidbody Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.rigidbody.html', 'snippet': 'Physics simulation operators for rigid body dynamics'},
                {'title': 'Physics World', 'url': 'https://docs.blender.org/api/current/bpy.types.RigidBodyWorld.html', 'snippet': 'Rigid body world settings and properties'}
            ],
            'mesh': [
                {'title': 'Mesh Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.mesh.html', 'snippet': 'Mesh creation and manipulation operators'},
                {'title': 'Mesh Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Mesh.html', 'snippet': 'Mesh data structure and properties'}
            ],
            'object': [
                {'title': 'Object Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.object.html', 'snippet': 'Object manipulation and creation operators'},
                {'title': 'Object Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Object.html', 'snippet': 'Object data structure and properties'}
            ],
            'light': [
                {'title': 'Light Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.object.html#bpy.ops.object.light_add', 'snippet': 'Light creation and manipulation'},
                {'title': 'Light Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Light.html', 'snippet': 'Light data structure and properties'}
            ],
            'material': [
                {'title': 'Material Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.material.html', 'snippet': 'Material creation and manipulation operators'},
                {'title': 'Material Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Material.html', 'snippet': 'Material data structure and properties'}
            ],
            'camera': [
                {'title': 'Camera Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.object.html#bpy.ops.object.camera_add', 'snippet': 'Camera creation and manipulation'},
                {'title': 'Camera Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Camera.html', 'snippet': 'Camera data structure and properties'}
            ],
            'scene': [
                {'title': 'Scene Operators', 'url': 'https://docs.blender.org/api/current/bpy.ops.scene.html', 'snippet': 'Scene management operators'},
                {'title': 'Scene Types', 'url': 'https://docs.blender.org/api/current/bpy.types.Scene.html', 'snippet': 'Scene data structure and properties'}
            ]
        }
        
        for keyword, sections in section_mapping.items():
            if keyword in query_lower:
                relevant_sections.extend(sections)
        
        return relevant_sections

    def _is_relevant_to_query(self, title: str, query: str) -> bool:
        """Check if a documentation title is relevant to the query"""
        query_lower = query.lower()
        title_lower = title.lower()
        
        # Keywords that indicate relevance
        relevant_keywords = ['physics', 'mesh', 'object', 'light', 'material', 'camera', 'scene', 'placement', 'generation']
        
        for keyword in relevant_keywords:
            if keyword in query_lower and keyword in title_lower:
                return True
        
        return False

    def search_documentation(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search both Blender and Infinigen documentation"""
        results = []
        
        # Search Blender documentation
        blender_results = self.search_blender_docs(query, max_results // 2)
        results.extend(blender_results)
        
        # Search Infinigen documentation
        infinigen_results = self.search_infinigen_docs(query, max_results // 2)
        results.extend(infinigen_results)
        
        return results[:max_results]
    
    def parse_instruction(self, instruction: str) -> Dict:
        """Parse user instruction and extract key information"""
        instruction_lower = instruction.lower()
        
        # Identify instruction type
        instruction_type = None
        for category, patterns in self.instruction_patterns.items():
            for pattern in patterns:
                if re.search(pattern, instruction_lower):
                    instruction_type = category
                    break
            if instruction_type:
                break
        
        # Extract object types
        object_types = ['cube', 'sphere', 'plane', 'cylinder', 'cone', 'torus', 'monkey']
        detected_objects = []
        for obj_type in object_types:
            if obj_type.lower() in instruction_lower:
                detected_objects.append(obj_type)
        
        # Extract position information
        position_match = re.search(r'location.*?\(([^)]+)\)|position.*?\(([^)]+)\)', instruction_lower)
        position = None
        if position_match:
            pos_str = position_match.group(1) or position_match.group(2)
            try:
                # Try to parse coordinates
                coords = [float(x.strip()) for x in pos_str.split(',')]
                if len(coords) >= 3:
                    position = tuple(coords[:3])
            except:
                position = None
        
        # Extract physics-related parameters
        physics_params = {}
        if re.search(r'gravity', instruction_lower):
            gravity_match = re.search(r'gravity.*?(\d+\.?\d*)', instruction_lower)
            if gravity_match:
                physics_params['gravity'] = float(gravity_match.group(1))
        
        if re.search(r'mass', instruction_lower):
            mass_match = re.search(r'mass.*?(\d+\.?\d*)', instruction_lower)
            if mass_match:
                physics_params['mass'] = float(mass_match.group(1))
        
        return {
            'instruction_type': instruction_type,
            'objects': detected_objects,
            'position': position,
            'physics_params': physics_params,
            'original_instruction': instruction
        }
    
    def search_knowledge_base(self, parsed_instruction: Dict) -> List[Dict]:
        """Search for relevant information in knowledge base"""
        results = []
        instruction_type = parsed_instruction.get('instruction_type')
        
        # Search relevant knowledge based on instruction type
        if instruction_type == 'physics_placement':
            results.extend(self.knowledge_base['bpy_physics']['apis'])
            results.extend(self.knowledge_base['infinigen_physics']['apis'])
        
        if instruction_type == 'object_creation':
            results.extend(self.knowledge_base['bpy_objects']['apis'])
        
        if instruction_type == 'scene_setup':
            results.extend(self.knowledge_base['infinigen_scene']['apis'])
        
        # If no specific type matches, search all relevant APIs
        if not results:
            for category in self.knowledge_base.values():
                results.extend(category['apis'])
        
        return results
    
    def generate_code_example(self, parsed_instruction: Dict, relevant_apis: List[Dict]) -> str:
        """Generate code example"""
        instruction_type = parsed_instruction.get('instruction_type')
        objects = parsed_instruction.get('objects', ['cube'])
        position = parsed_instruction.get('position', (0, 0, 0))
        physics_params = parsed_instruction.get('physics_params', {})
        
        code_lines = ["import bpy", "import bmesh", ""]
        
        # Generate corresponding code based on instruction type
        if instruction_type == 'physics_placement':
            code_lines.extend([
                "# Setup physics world",
                "bpy.ops.rigidbody.world_add()",
                ""
            ])
            
            # Set gravity
            gravity = physics_params.get('gravity', -9.81)
            code_lines.extend([
                "# Set gravity",
                f"bpy.context.scene.rigidbody_world.gravity = (0, 0, {gravity})",
                ""
            ])
            
            # Create objects
            for obj_name in objects:
                if 'cube' in obj_name.lower():
                    code_lines.extend([
                        "# Create cube",
                        f"bpy.ops.mesh.primitive_cube_add(size=2, location={position})",
                        ""
                    ])
                elif 'sphere' in obj_name.lower():
                    code_lines.extend([
                        "# Create sphere",
                        f"bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location={position})",
                        ""
                    ])
                elif 'plane' in obj_name.lower():
                    code_lines.extend([
                        "# Create plane",
                        f"bpy.ops.mesh.primitive_plane_add(size=10, location={position})",
                        ""
                    ])
            
            # Add rigid body physics
            mass = physics_params.get('mass', 1.0)
            code_lines.extend([
                "# Add rigid body physics to object",
                f"bpy.ops.rigidbody.object_add(type='ACTIVE')",
                f"bpy.context.object.rigid_body.mass = {mass}",
                ""
            ])
            
        elif instruction_type == 'object_creation':
            for obj_name in objects:
                if 'cube' in obj_name.lower():
                    code_lines.extend([
                        "# Create cube",
                        f"bpy.ops.mesh.primitive_cube_add(size=2, location={position})",
                        ""
                    ])
                elif 'sphere' in obj_name.lower():
                    code_lines.extend([
                        "# Create sphere",
                        f"bpy.ops.mesh.primitive_uv_sphere_add(radius=1, location={position})",
                        ""
                    ])
        
        # Add comments explaining used APIs
        if relevant_apis:
            code_lines.extend([
                "# Related API documentation:",
            ])
            for api in relevant_apis[:3]:  # Show only first 3
                code_lines.append(f"# {api['name']}: {api['description']}")
        
        return "\n".join(code_lines)
    
    def enhanced_generation(self, instruction: str) -> str:
        """Use OpenAI for enhanced code generation (if available)"""
        if not self.openai_client:
            return "OpenAI API not available for enhanced generation"
        
        try:
            parsed = self.parse_instruction(instruction)
            relevant_apis = self.search_knowledge_base(parsed)
            
            # Search official documentation for additional context
            doc_results = self.search_documentation(instruction, max_results=5)
            doc_context = ""
            if doc_results:
                doc_context = "\n\nOfficial Documentation References:\n"
                for result in doc_results:
                    doc_context += f"- {result['title']} ({result['source']}): {result['snippet']}\n"
                    doc_context += f"  URL: {result['url']}\n"
            
            # Build prompt
            api_info = "\n".join([f"- {api['name']}: {api['description']}" for api in relevant_apis[:5]])
            
            prompt = f"""
Based on the following user instruction, relevant API information, and official documentation references, generate Blender Python code:

User instruction: {instruction}

Relevant APIs from knowledge base:
{api_info}
{doc_context}

Requirements:
1. Generate complete, executable Blender Python code
2. Include necessary import statements
3. Add detailed English comments
4. Ensure code follows Blender Python API best practices
5. Use the most current and accurate API methods from the official documentation
6. Reference the official Blender Python API documentation at https://docs.blender.org/api/current/index.html

Please return only the code, no additional explanations.
"""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logging.error(f"Enhanced generation failed: {e}")
            return f"Enhanced generation failed: {str(e)}"
    
    def rag_query(self, instruction: str, use_enhanced: bool = False, use_doc_search: bool = True) -> Dict:
        """Main RAG query function"""
        try:
            # Parse instruction
            parsed_instruction = self.parse_instruction(instruction)
            
            # Search relevant knowledge
            relevant_apis = self.search_knowledge_base(parsed_instruction)
            
            # Search official documentation if requested
            doc_results = []
            if use_doc_search:
                doc_results = self.search_documentation(instruction, max_results=5)
            
            # Generate code example
            if use_enhanced and self.openai_client:
                code_example = self.enhanced_generation(instruction)
            else:
                code_example = self.generate_code_example(parsed_instruction, relevant_apis)
            
            return {
                'status': 'success',
                'instruction': instruction,
                'parsed_instruction': parsed_instruction,
                'relevant_apis': relevant_apis,
                'doc_results': doc_results,
                'code_example': code_example,
                'generation_method': 'enhanced' if use_enhanced else 'template'
            }
            
        except Exception as e:
            logging.error(f"RAG query failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'instruction': instruction
            }


# Global instance
_rag_tool = None


def initialize_rag_tool(openai_api_key: str = None):
    """Initialize RAG tool"""
    global _rag_tool
    _rag_tool = BlenderInfinigenRAG(openai_api_key)
    return _rag_tool


def rag_query(instruction: str, use_enhanced: bool = False, use_doc_search: bool = True) -> Dict:
    """RAG query interface"""
    global _rag_tool
    if _rag_tool is None:
        _rag_tool = BlenderInfinigenRAG()
    
    return _rag_tool.rag_query(instruction, use_enhanced, use_doc_search)


# MCP tool interface
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("rag-tool")

@mcp.tool()
def rag_query_tool(instruction: str, use_enhanced: bool = False, use_doc_search: bool = True) -> dict:
    """
    Search for bpy and Infinigen related documentation and generate code examples based on input instruction
    
    Args:
        instruction: User instruction, e.g., "place an object following physics laws"
        use_enhanced: Whether to use OpenAI enhanced generation (requires OpenAI API key)
        use_doc_search: Whether to search official Blender and Infinigen documentation
        
    Returns:
        dict: Dictionary containing parsed results, relevant APIs, documentation results, and generated code
    """
    return rag_query(instruction, use_enhanced, use_doc_search)

@mcp.tool()
def search_blender_docs_tool(query: str, max_results: int = 5) -> dict:
    """
    Search Blender Python API documentation for relevant information
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        dict: Blender documentation search results
    """
    try:
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG()
        
        results = _rag_tool.search_blender_docs(query, max_results)
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def search_infinigen_docs_tool(query: str, max_results: int = 5) -> dict:
    """
    Search Infinigen documentation for relevant information
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        dict: Infinigen documentation search results
    """
    try:
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG()
        
        results = _rag_tool.search_infinigen_docs(query, max_results)
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def search_documentation_tool(query: str, max_results: int = 5) -> dict:
    """
    Search both Blender and Infinigen documentation for relevant information
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        dict: Combined documentation search results from both sources
    """
    try:
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG()
        
        results = _rag_tool.search_documentation(query, max_results)
        return {
            "status": "success",
            "query": query,
            "results": results
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def initialize_rag_tool(openai_api_key: str = None) -> dict:
    """
    Initialize RAG tool
    
    Args:
        openai_api_key: OpenAI API key (optional)
        
    Returns:
        dict: Initialization result
    """
    try:
        global _rag_tool
        _rag_tool = BlenderInfinigenRAG(openai_api_key)
        return {"status": "success", "message": "RAG tool initialized successfully"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    """Run MCP server or test the RAG tool when --test is provided"""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Test instruction (Chinese as requested)
        test_instruction = "将床移动到房间角落"
        # Ensure tool is initialized
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG()
        # Run query with official doc search enabled
        result = _rag_tool.rag_query(test_instruction, use_enhanced=False, use_doc_search=True)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
        # Default: run as MCP server
    else:
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
