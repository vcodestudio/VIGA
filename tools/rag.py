# rag.py
import os
import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import uuid

# tool_configs for agent (only the function w/ @mcp.tools)
tool_configs = [
    {
        "type": "function",
        "function": {
            "name": "rag_query",
            "description": "Search knowledge base using vector similarity",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "description": "Maximum number of results to return"},
                    "domain_filter": {"type": "string", "description": "Filter by domain ('blender' or 'infinigen')"}
                }
            }
        }
    }
]

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None


class BlenderInfinigenRAG:
    """RAG tool: Vector-based search through knowledge.jsonl for Blender and Infinigen documentation"""
    
    def __init__(self, openai_api_key: str = None, knowledge_file: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.knowledge_file = knowledge_file or "tools/rag_knowledge/knowledge.jsonl"
        
        # Initialize OpenAI client
        if self.openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        else:
            self.openai_client = None
            if not OPENAI_AVAILABLE:
                logging.warning("OpenAI library not available. Enhanced generation disabled.")
            elif not self.openai_api_key:
                logging.warning("OpenAI API key not provided. Enhanced generation disabled.")
        
        # Initialize vector database
        self.vector_db = None
        self.embedding_model = None
        self._init_vector_db()
        
        # Instruction pattern matching for code generation
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
    
    def _init_vector_db(self):
        """Initialize vector database for knowledge search"""
        if not CHROMADB_AVAILABLE:
            logging.warning("ChromaDB not available. Falling back to simple text search.")
            return
        
        try:
            # Initialize ChromaDB client
            self.vector_db = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory="./chroma_db"
            ))
            
            # Initialize embedding model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            elif self.openai_client:
                # Use OpenAI embeddings as fallback
                self.embedding_model = "openai"
            else:
                logging.warning("No embedding model available. Using simple text search.")
                return
            
            # Load knowledge from JSONL file
            self._load_knowledge_to_vector_db()
            
        except Exception as e:
            logging.error(f"Failed to initialize vector database: {e}")
            self.vector_db = None
    
    def _load_knowledge_to_vector_db(self):
        """Load knowledge from JSONL file into vector database"""
        if not self.vector_db or not os.path.exists(self.knowledge_file):
            return
        
        try:
            # Check if collection already exists
            collection_name = "blender_infinigen_knowledge"
            try:
                collection = self.vector_db.get_collection(collection_name)
                logging.info("Using existing knowledge collection")
                return
            except:
                collection = self.vector_db.create_collection(collection_name)
                logging.info("Created new knowledge collection")
            
            # Load and process knowledge entries
            documents = []
            metadatas = []
            ids = []
            
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line.strip())
                        
                        # Create searchable text from entry
                        searchable_text = self._create_searchable_text(entry)
                        
                        documents.append(searchable_text)
                        metadatas.append({
                            'id': entry.get('id', str(uuid.uuid4())),
                            'domain': entry.get('domain', ''),
                            'title': entry.get('title', ''),
                            'url': entry.get('url', ''),
                            'version': entry.get('version', ''),
                            'tags': ','.join(entry.get('tags', [])),
                            'updated': entry.get('updated', ''),
                            'source_type': entry.get('source_type', ''),
                            'section_path': ','.join(entry.get('section_path', [])),
                            'line_number': line_num
                        })
                        ids.append(entry.get('id', str(uuid.uuid4())))
                        
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse line {line_num}: {e}")
                        continue
            
            if documents:
                # Add documents to collection
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                logging.info(f"Loaded {len(documents)} knowledge entries into vector database")
            
        except Exception as e:
            logging.error(f"Failed to load knowledge to vector database: {e}")
    
    def _create_searchable_text(self, entry: Dict) -> str:
        """Create searchable text from knowledge entry"""
        parts = []
        
        # Add title
        if entry.get('title'):
            parts.append(f"Title: {entry['title']}")
        
        # Add content summary
        if entry.get('content_summary'):
            parts.append(f"Content: {entry['content_summary']}")
        
        # Add tags
        if entry.get('tags'):
            parts.append(f"Tags: {', '.join(entry['tags'])}")
        
        # Add section path
        if entry.get('section_path'):
            parts.append(f"Section: {' > '.join(entry['section_path'])}")
        
        # Add domain
        if entry.get('domain'):
            parts.append(f"Domain: {entry['domain']}")
        
        return "\n".join(parts)
    
    def search_vector_knowledge(self, query: str, max_results: int = 5, domain_filter: str = None) -> List[Dict]:
        """Search knowledge using vector similarity"""
        if not self.vector_db:
            return self._fallback_text_search(query, max_results, domain_filter)
        
        try:
            collection = self.vector_db.get_collection("blender_infinigen_knowledge")
            
            # Apply domain filter if specified
            where_clause = None
            if domain_filter:
                where_clause = {"domain": domain_filter}
            
            # Perform vector search
            results = collection.query(
                query_texts=[query],
                n_results=max_results,
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {}
                    distance = results['distances'][0][i] if results['distances'] and results['distances'][0] else 0
                    
                    formatted_results.append({
                        'title': metadata.get('title', 'Unknown'),
                        'url': metadata.get('url', ''),
                        'snippet': doc[:200] + '...' if len(doc) > 200 else doc,
                        'source': metadata.get('domain', 'unknown'),
                        'similarity': 1 - distance,  # Convert distance to similarity
                        'tags': metadata.get('tags', ''),
                        'section_path': metadata.get('section_path', ''),
                        'content': doc
                    })
            
            return formatted_results
            
        except Exception as e:
            logging.error(f"Vector search failed: {e}")
            return self._fallback_text_search(query, max_results, domain_filter)
    
    def _fallback_text_search(self, query: str, max_results: int = 5, domain_filter: str = None) -> List[Dict]:
        """Fallback text search when vector search is not available"""
        if not os.path.exists(self.knowledge_file):
            return []
        
        results = []
        query_lower = query.lower()
        
        try:
            with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line.strip())
                        
                        # Apply domain filter
                        if domain_filter and entry.get('domain') != domain_filter:
                            continue
                        
                        # Calculate relevance score
                        score = 0
                        text_to_search = f"{entry.get('title', '')} {entry.get('content_summary', '')} {' '.join(entry.get('tags', []))}"
                        text_lower = text_to_search.lower()
                        
                        # Simple keyword matching
                        query_words = query_lower.split()
                        for word in query_words:
                            if word in text_lower:
                                score += 1
                        
                        if score > 0:
                            results.append({
                                'title': entry.get('title', 'Unknown'),
                                'url': entry.get('url', ''),
                                'snippet': entry.get('content_summary', '')[:200] + '...' if len(entry.get('content_summary', '')) > 200 else entry.get('content_summary', ''),
                                'source': entry.get('domain', 'unknown'),
                                'similarity': score / len(query_words),  # Normalized score
                                'tags': ','.join(entry.get('tags', [])),
                                'section_path': ' > '.join(entry.get('section_path', [])),
                                'content': entry.get('content_summary', '')
                            })
                        
                        if len(results) >= max_results * 2:  # Get more than needed for sorting
                            break
                            
                    except json.JSONDecodeError:
                        continue
            
            # Sort by similarity score and return top results
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            logging.error(f"Fallback text search failed: {e}")
            return []

    def search_blender_docs(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Blender documentation in knowledge base"""
        return self.search_vector_knowledge(query, max_results, domain_filter='blender')

    def search_infinigen_docs(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search Infinigen documentation in knowledge base"""
        return self.search_vector_knowledge(query, max_results, domain_filter='infinigen')

    def search_documentation(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search both Blender and Infinigen documentation in knowledge base"""
        # Search both domains and combine results
        blender_results = self.search_blender_docs(query, max_results // 2 or 1)
        infinigen_results = self.search_infinigen_docs(query, max_results - len(blender_results))
        
        # Combine and sort by similarity
        all_results = blender_results + infinigen_results
        all_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        return all_results[:max_results]

    # ---------------------------
    # Knowledge base search
    # ---------------------------
    
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
        """Search for relevant information in knowledge base using vector search"""
        instruction = parsed_instruction.get('original_instruction', '')
        if not instruction:
            return []
        
        # Use vector search to find relevant knowledge
        vector_results = self.search_vector_knowledge(instruction, max_results=10)
        
        # Convert vector results to API-like format for compatibility
        api_results = []
        for result in vector_results:
            # Extract potential API information from the content
            content = result.get('content', '')
            title = result.get('title', '')
            
            # Try to find API names and descriptions in the content
            api_matches = re.findall(r'(bpy\.\w+(?:\.\w+)*)', content)
            if api_matches:
                for api_name in set(api_matches):  # Remove duplicates
                    api_results.append({
                        'name': api_name,
                        'description': f"Found in {title}: {content[:100]}...",
                        'example': '',
                        'use_case': result.get('tags', ''),
                        'source': result.get('source', ''),
                        'similarity': result.get('similarity', 0),
                        'url': result.get('url', '')
                    })
            else:
                # If no specific APIs found, create a general entry
                api_results.append({
                    'name': title,
                    'description': content[:200] + '...' if len(content) > 200 else content,
                    'example': '',
                    'use_case': result.get('tags', ''),
                    'source': result.get('source', ''),
                    'similarity': result.get('similarity', 0),
                    'url': result.get('url', '')
                })
        
        # Sort by similarity and return top 5
        api_results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        return api_results[:5]
    
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
            
            # Search knowledge base for additional context
            doc_results = self.search_documentation(instruction, max_results=5)
            doc_context = ""
            if doc_results:
                doc_context = "\n\nKnowledge Base References:\n"
                for result in doc_results:
                    doc_context += f"- {result['title']} ({result['source']}): {result['snippet']}\n"
                    if result.get('url'):
                        doc_context += f"  URL: {result['url']}\n"
                    if result.get('similarity'):
                        doc_context += f"  Relevance: {result['similarity']:.2f}\n"
            
            # Build prompt
            api_info = "\n".join([f"- {api['name']}: {api['description']}" for api in relevant_apis[:5]])
            
            prompt = f"""
Based on the following user instruction, relevant API information, and knowledge base references, generate Blender Python code:

User instruction: {instruction}

Relevant APIs from knowledge base:
{api_info}
{doc_context}

Requirements:
1. Generate complete, executable Blender Python code
2. Include necessary import statements
3. Add detailed English comments
4. Ensure code follows Blender Python API best practices
5. Use the most current and accurate API methods from the knowledge base
6. Focus on practical, working examples

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
        """Main RAG query function using vector search"""
        try:
            # Parse instruction
            parsed_instruction = self.parse_instruction(instruction)
            
            # Search relevant knowledge using vector search
            relevant_apis = self.search_knowledge_base(parsed_instruction)
            
            # Search documentation using vector search if requested
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
                'generation_method': 'enhanced' if use_enhanced else 'template',
                'search_method': 'vector' if self.vector_db else 'text'
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
def initialize(args: dict) -> dict:
    """
    Initialize RAG tool with vector database
    
    Args:
        openai_api_key: OpenAI API key (optional)
        knowledge_file: Path to knowledge.jsonl file (optional)
        
    Returns:
        dict: Initialization result
    """
    try:
        global _rag_tool
        _rag_tool = BlenderInfinigenRAG(
            openai_api_key=args.get("openai_api_key"),
            knowledge_file=args.get("knowledge_file", "tools/rag_knowledge/knowledge.jsonl")
        )
        return {
            "status": "success", 
            "message": "RAG tool initialized successfully with vector database",
            "vector_db_available": _rag_tool.vector_db is not None,
            "embedding_model": "sentence-transformers" if SENTENCE_TRANSFORMERS_AVAILABLE else ("openai" if _rag_tool.openai_client else "none")
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def search_knowledge_vector(query: str, max_results: int = 5, domain_filter: str = None) -> dict:
    """
    Search knowledge base using vector similarity
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        domain_filter: Filter by domain ('blender' or 'infinigen')
        
    Returns:
        dict: Vector search results with similarity scores
    """
    try:
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG()
        
        results = _rag_tool.search_vector_knowledge(query, max_results, domain_filter)
        return {
            "status": "success",
            "query": query,
            "results": results,
            "search_method": "vector" if _rag_tool.vector_db else "text_fallback"
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    """Run MCP server or test the RAG tool when --test is provided"""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Ensure tool is initialized (optionally with OPENAI_API_KEY env)
        global _rag_tool
        if _rag_tool is None:
            _rag_tool = BlenderInfinigenRAG(os.getenv("OPENAI_API_KEY"))

        # Test 1: Initialize with vector database
        print("[test:initialize] Initializing RAG tool with vector database...")
        init_args = {
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "knowledge_file": "tools/rag_knowledge/knowledge.jsonl"
        }
        init_result = initialize(init_args)
        print("[test:initialize]", json.dumps(init_result, ensure_ascii=False, indent=2))

        # Test 2: Vector search - general query
        test_instruction = os.getenv("RAG_TEST_INSTRUCTION", "Place a cube on a plane and set physics")
        print(f"[test:vector_search] Testing vector search with: {test_instruction}")
        vector_results = _rag_tool.search_vector_knowledge(test_instruction, max_results=3)
        print("[test:vector_search]", json.dumps(vector_results, ensure_ascii=False, indent=2))

        # Test 3: Domain-specific search - Blender
        print("[test:blender_search] Testing Blender-specific search...")
        blender_results = _rag_tool.search_blender_docs("bpy.ops.mesh.primitive_cube_add", max_results=3)
        print("[test:blender_search]", json.dumps(blender_results, ensure_ascii=False, indent=2))

        # Test 4: Domain-specific search - Infinigen
        print("[test:infinigen_search] Testing Infinigen-specific search...")
        infinigen_results = _rag_tool.search_infinigen_docs("placement physics", max_results=3)
        print("[test:infinigen_search]", json.dumps(infinigen_results, ensure_ascii=False, indent=2))

        # Test 5: Full RAG query with vector search
        print("[test:rag_query] Testing full RAG query...")
        rag_result = _rag_tool.rag_query(test_instruction, use_enhanced=False, use_doc_search=True)
        print("[test:rag_query]", json.dumps(rag_result, ensure_ascii=False, indent=2))

        # Test 6: MCP tool interface - search_knowledge_vector
        print("[test:mcp_vector_search] Testing MCP vector search tool...")
        mcp_vector_result = search_knowledge_vector("rigid body physics", max_results=3, domain_filter="blender")
        print("[test:mcp_vector_search]", json.dumps(mcp_vector_result, ensure_ascii=False, indent=2))
        return
    else:
        # Default: run as MCP server
        mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
